from typing import Callable, List, NamedTuple, Optional, Tuple

import torch
import torch.nn.functional as F
import torch_geometric
from torch import Tensor

from torch_geometric.utils import coalesce, scatter, softmax, remove_self_loops, cumsum
import torch_scatter

class UnpoolInfo(NamedTuple):
    edge_index: Tensor
    cluster: Tensor
    batch: Tensor
    new_edge_score: Tensor


class XPooling(torch.nn.Module):
    """
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
            add_to_edge_score: float = 0.0,
            mlp1=None,
            mlp2=None,
            alpha=0.995,
            merge=False
    ):
        super().__init__()
        self.in_channels = in_channels
        if edge_score_method is None:
            edge_score_method = self.compute_edge_score_sigmoid
        self.compute_edge_score = edge_score_method
        self.add_to_edge_score = add_to_edge_score
        self.lin = torch.nn.Linear(2 * in_channels, 1)
        #self.lin.weight.data.fill_(1)
        #self.lin.bias.data.fill_(0)

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
        edge_index, _ = remove_self_loops(edge_index)
        #torch_geometric.seed_everything(0)
        #x = torch.randn_like(x)
        if scores is None:
            e = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1)
            e = self.lin(e).view(-1)
            e = self.compute_edge_score(e, edge_index, x.size(0))
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
        mask = torch.ones(x.size(0), dtype=torch.bool)
        new_batch = []
        edge_batch = batch[edge_index[0]]
        i = 0
        if len(edge_score) > 0:
            node_mask = torch.ones(x.size(0), dtype=torch.bool)
            edge_batch = batch[edge_index[0]]
            edge_idxs = torch_scatter.scatter_max(edge_score, edge_batch, dim=0)[1]
            x_new = x[edge_index[0, edge_idxs]] + x[edge_index[1, edge_idxs]]
            node_mask[edge_index[:, edge_idxs].view(-1)] = False
            batch_new = edge_batch[edge_idxs]
            new_x = torch.cat([x_new, x[node_mask]], dim=0)
            new_batch = torch.cat([batch_new, batch[node_mask]])
            batch_nums = torch.arange(0, len(edge_idxs), device=cluster.device)
            cluster[edge_index[0, edge_idxs]] = batch_nums
            cluster[edge_index[1, edge_idxs]] = batch_nums
            cluster[node_mask] = torch.arange(len(edge_idxs), node_mask.sum() + len(edge_idxs), device=cluster.device)
            new_edge_index = coalesce(cluster[edge_index], num_nodes=new_x.size(0))
        else:
            new_x = x
            new_batch = batch
            new_edge_index = edge_index

        return new_x, new_edge_index, new_batch, None

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.in_channels})'
