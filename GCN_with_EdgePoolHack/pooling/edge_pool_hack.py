from typing import Callable, List, NamedTuple, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.utils import coalesce, scatter, softmax


class UnpoolInfo(NamedTuple):
    edge_index: Tensor
    cluster: Tensor
    batch: Tensor
    new_edge_score: Tensor


class EdgePoolingHack(torch.nn.Module):
    r"""The edge pooling operator from the `"Towards Graph Pooling by Edge
    Contraction" <https://graphreason.github.io/papers/17.pdf>`__ and
    `"Edge Contraction Pooling for Graph Neural Networks"
    <https://arxiv.org/abs/1905.10990>`__ papers.

    In short, a score is computed for each edge.
    Edges are contracted iteratively according to that score unless one of
    their nodes has already been part of a contracted edge.

    To duplicate the configuration from the `"Towards Graph Pooling by Edge
    Contraction" <https://graphreason.github.io/papers/17.pdf>`__ paper, use
    either :func:`EdgePooling.compute_edge_score_softmax`
    or :func:`EdgePooling.compute_edge_score_tanh`, and set
    :obj:`add_to_edge_score` to :obj:`0.0`.

    To duplicate the configuration from the `"Edge Contraction Pooling for
    Graph Neural Networks" <https://arxiv.org/abs/1905.10990>`__ paper,
    set :obj:`dropout` to :obj:`0.2`.

    Args:
        in_channels (int): Size of each input sample.
        edge_score_method (callable, optional): The function to apply
            to compute the edge score from raw edge scores. By default,
            this is the softmax over all incoming edges for each node.
            This function takes in a :obj:`raw_edge_score` tensor of shape
            :obj:`[num_nodes]`, an :obj:`edge_index` tensor and the number of
            nodes :obj:`num_nodes`, and produces a new tensor of the same size
            as :obj:`raw_edge_score` describing normalized edge scores.
            Included functions are
            :func:`EdgePooling.compute_edge_score_softmax`,
            :func:`EdgePooling.compute_edge_score_tanh`, and
            :func:`EdgePooling.compute_edge_score_sigmoid`.
            (default: :func:`EdgePooling.compute_edge_score_softmax`)
        dropout (float, optional): The probability with
            which to drop edge scores during training. (default: :obj:`0.0`)
        add_to_edge_score (float, optional): A value to be added to each
            computed edge score. Adding this greatly helps with unpooling
            stability. (default: :obj:`0.5`)
        mlp1 should be an MLP which accepts input of dimension in_channels
        mlp2 should be an MLP which accepts the output dimension of mlp1 as input dimension.
            The output dimension of mlp2 will be the new node representation dimension.
    """

    def __init__(
            self,
            in_channels: int,
            edge_score_method: Optional[Callable] = None,
            dropout: Optional[float] = 0.0,
            add_to_edge_score: float = 0.5,
            mlp1=None,
            mlp2=None,
            learnable=True
    ):
        super().__init__()
        self.in_channels = in_channels
        if edge_score_method is None:
            edge_score_method = self.compute_edge_score_softmax
        self.compute_edge_score = edge_score_method
        self.add_to_edge_score = add_to_edge_score
        self.dropout = dropout
        self.lin = torch.nn.Sequential(torch.nn.Linear(2 * in_channels, 2 * in_channels), torch.nn.ReLU(),
                                       torch.nn.Linear(2 * in_channels, 1))

        self.mlp1 = mlp1
        self.mlp2 = mlp2
        self.learnable = learnable

    @staticmethod
    def compute_edge_score_softmax(
            raw_edge_score: Tensor,
            edge_index: Tensor,
            num_nodes: int,
    ) -> Tensor:
        r"""Normalizes edge scores via softmax application."""
        return softmax(raw_edge_score, edge_index[1], num_nodes=num_nodes)

    @staticmethod
    def compute_edge_score_tanh(
            raw_edge_score: Tensor,
            edge_index: Optional[Tensor] = None,
            num_nodes: Optional[int] = None,
    ) -> Tensor:
        r"""Normalizes edge scores via hyperbolic tangent application."""
        return torch.tanh(raw_edge_score)

    @staticmethod
    def compute_edge_score_sigmoid(
            raw_edge_score: Tensor,
            edge_index: Optional[Tensor] = None,
            num_nodes: Optional[int] = None,
    ) -> Tensor:
        r"""Normalizes edge scores via sigmoid application."""
        return torch.sigmoid(raw_edge_score)

    def forward(
            self,
            x: Tensor,
            edge_index: Tensor,
            batch: Tensor,
            scores: Tensor = None,
    ) -> Tuple[Tensor, Tensor, Tensor, UnpoolInfo]:
        r"""Forward pass.

        Args:
            x (torch.Tensor): The node features.
            edge_index (torch.Tensor): The edge indices.
            batch (torch.Tensor): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each node to a specific example.

        Return types:
            * **x** *(torch.Tensor)* - The pooled node features.
            * **edge_index** *(torch.Tensor)* - The coarsened edge indices.
            * **batch** *(torch.Tensor)* - The coarsened batch vector.
            * **unpool_info** *(UnpoolInfo)* - Information that is
              consumed by :func:`EdgePooling.unpool` for unpooling.
        """
        if scores is None:
            e = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1)
            e = self.lin(e).view(-1)
            e = F.dropout(e, p=self.dropout, training=self.training)
            # e = self.compute_edge_score(e, edge_index, x.size(0))
            e = e + self.add_to_edge_score
        else:
            e = scores

        x, edge_index, batch, unpool_info = self._merge_edges(
            x, edge_index, batch, e)

        return x, edge_index, batch, unpool_info

    def _merge_edges(
            self,
            x: Tensor,
            edge_index: Tensor,
            batch: Tensor,
            edge_score: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, UnpoolInfo]:

        cluster = torch.empty_like(batch)
        x = self.mlp1(x)

        mask = torch.ones(x.size(0), dtype=torch.bool)
        new_batch = []
        new_x = []
        edge_batch = batch[edge_index[0]]
        i = 0
        for batch_id in range(batch.max() + 1):
            batch_edges = edge_batch == batch_id
            scores = edge_score[batch_edges]
            if self.learnable:
                if self.training:
                    one_hot = torch.nn.functional.gumbel_softmax(scores, hard=True)
                    max_idx = one_hot.max(dim=0)[1]
                else:
                    max_idx = torch.argmax(scores)
                    one_hot = torch.zeros_like(scores).scatter_(0, max_idx, 1.)
                    new_x.append((x[edge_index[:, batch_edges]] * one_hot.view(1, -1, 1)).sum(dim=[0, 1]))
            else:
                max_idx = torch.argmax(scores)
            edge = edge_index[:, batch_edges][:, max_idx]
            cluster[edge[0]] = i
            cluster[edge[1]] = i
            mask[edge[0]] = False
            mask[edge[1]] = False
            new_batch.append(batch_id)
            i += 1

        j = int(mask.sum())
        cluster[mask] = torch.arange(i, i + j, device=x.device)
        i += j
        new_batch = torch.cat([torch.LongTensor(new_batch).to(batch.device), batch[mask]])

        if self.learnable:
            new_x = torch.stack(new_x)
            new_x = torch.cat([new_x, x[mask]], dim=0)
        else:
            new_x = scatter(x, cluster, dim=0, dim_size=i, reduce='sum')

        # We compute the new features as an addition of the old ones after transformation and apply another mlp
        # X multiset
        # f(X) = g(\sum_{x\in X} h(x))

        new_x = self.mlp2(new_x)
        # new_edge_score = edge_score[new_edge_indices]
        # if int(mask.sum()) > 0:
        #    remaining_score = x.new_ones(
        #        (new_x.size(0) - len(new_edge_indices),))
        #    new_edge_score = torch.cat([new_edge_score, remaining_score])
        # new_x = new_x * new_edge_score.view(-1, 1)

        new_edge_index = coalesce(cluster[edge_index], num_nodes=new_x.size(0))
        # new_batch = x.new_empty(new_x.size(0), dtype=torch.long)
        # new_batch = new_batch.scatter_(0, cluster, batch)

        # unpool_info = UnpoolInfo(edge_index=edge_index, cluster=cluster,
        #                         batch=batch, new_edge_score=new_edge_score)

        # print('X:',x.shape, new_x.shape, torch.max(cluster), torch.max(new_edge_index), i, 'Edges:', edge_index.shape, new_edge_index.shape, print(len(cluster)), new_batch.max())

        return new_x, new_edge_index, new_batch, None  # unpool_info

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.in_channels})'
