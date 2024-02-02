from typing import Callable, List, NamedTuple, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.utils import coalesce, scatter, softmax, remove_self_loops, cumsum


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
            add_to_edge_score: float = 0.5,
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
        self.dropout = dropout
        self.lin = torch.nn.Sequential(torch.nn.Linear(2 * in_channels, 2 * in_channels), torch.nn.ReLU(),
                                       torch.nn.Linear(2 * in_channels, 1))

        self.mlp1 = mlp1
        self.mlp2 = mlp2
        self.alpha = alpha
        self.epsilon = 1e-5
        self.merge = merge        

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
        if scores is None:
            e = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1)
            e = self.lin(e).view(-1)
            e = F.dropout(e, p=self.dropout, training=self.training)
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
        x = self.mlp1(x)
        mask = torch.ones(x.size(0), dtype=torch.bool)
        new_batch = []
        edge_batch = batch[edge_index[0]]
        i = 0
        if len(edge_score) > 0:
            if self.merge == 'max':
                node_mask = torch.ones(x.size(0), dtype=torch.bool)
                edge_batch = batch[edge_index[0]]
                num_edges = scatter(edge_batch.new_ones(edge_score.size(0)), edge_batch, reduce='sum')
                k = num_edges.new_full((num_edges.size(0),), 1)
                edge_scores, edge_scores_perm = torch.sort(edge_score.view(-1), descending=True)
                edge_batch_sorted = edge_batch[edge_scores_perm]
                edge_batch_sorted, edge_batch_perm = torch.sort(edge_batch_sorted, descending=False, stable=True)
                arange = torch.arange(edge_scores.size(0), dtype=torch.long, device=x.device)
                ptr = cumsum(num_edges)
                edge_batched_arange = arange - ptr[edge_batch_sorted]
                mask = edge_batched_arange < k[edge_batch_sorted]
                edge_idxs = edge_scores_perm[edge_batch_perm[mask]]
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
                for batch_id in range(batch.max() + 1):
                    is_in_batch = edge_batch == batch_id
                    batch_scores = edge_score[is_in_batch]
                    if len(batch_scores) > 0:
                        max_score = batch_scores.max()
                        min_score = batch_scores.min()
                        threshold = min_score + (max_score - min_score) * self.alpha - self.epsilon
                        edges_to_pool = batch_scores > threshold
                        edges_in_batch = edge_index[:, is_in_batch][:, edges_to_pool]
                        for edge in edges_in_batch.t():
                            if not mask[edge[0]]:
                                if self.merge == 'combine':
                                    if (not mask[edge[1]]) and (cluster[edge[0]] != cluster[edge[1]]):
                                        other_cluster_id = torch.max(cluster[edge[1]], cluster[edge[0]])
                                        reassign = cluster == other_cluster_id
                                        cluster[reassign] = torch.min(cluster[edge[0]], cluster[edge[1]])
                                        too_high = cluster > other_cluster_id
                                        cluster[too_high] -= 1
                                        i -= 1
                                        del new_batch[other_cluster_id]
                                    else:
                                        cluster[edge[1]] = cluster[edge[0]]
                                        mask[edge[1]] = False
                                else:
                                    continue
                            elif not mask[edge[1]]:
                                if self.merge == 'combine':
                                    cluster[edge[0]] = cluster[edge[1]]
                                    mask[edge[0]] = False
                                else:
                                    continue
                            else:
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

                new_x = scatter(x, cluster, dim=0, dim_size=i, reduce='sum')
                new_edge_index = coalesce(cluster[edge_index], num_nodes=new_x.size(0))
        else:
            new_x = x
            new_batch = batch
            new_edge_index = edge_index

        new_x = self.mlp2(new_x)
        return new_x, new_edge_index, new_batch, None

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.in_channels})'
