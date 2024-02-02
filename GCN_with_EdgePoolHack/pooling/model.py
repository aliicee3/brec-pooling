import torch
from torch.nn import Linear, Parameter, Module
from torch_geometric.nn import GCNConv, GINConv, GATConv
from torch_geometric.nn.pool import global_add_pool, TopKPooling, EdgePooling
from torch_geometric.utils import add_self_loops, degree
import torch_geometric as pyg
from pooling.XPooling import XPooling


class DiffPool(torch.nn.Module):
    def __init__(self, in_channels, num_clusters):
        super().__init__()
        mlp = torch.nn.Sequential(torch.nn.Linear(in_channels, in_channels), torch.nn.ReLU(),
                                   torch.nn.Linear(in_channels, num_clusters))
        self.conv = GINConv(mlp)
        self.num_clusters = num_clusters

    def forward(self, x, edge_idx, batch):
        x_dense, _ = pyg.utils.to_dense_batch(x, batch)
        s, _ = pyg.utils.to_dense_batch(self.conv(x, edge_idx), batch)
        adj = pyg.utils.to_dense_adj(edge_idx, batch)
        x_dense, adj, _, _ = pyg.nn.dense_diff_pool(x_dense, adj, s)
        edge_idx = pyg.utils.dense_to_sparse(adj)[0]
        batch_count = torch.max(batch).item() + 1
        batch = torch.arange(batch_count).repeat_interleave(self.num_clusters).to(batch.device)
        return x_dense.view(self.num_clusters*batch_count, -1), edge_idx, batch, None

class GINandPool(torch.nn.Module):
    
    POOLING_OPTIONS = ['xp', 'edge_pool', 'topk', 'none', 'sag', 'asa', 'diff_pool']
    def __init__(self, in_channels, hidden_dim, out_channels, pool='topk', num_blocks=3, num_layers=4, conv_type='gin',
                 merge=False, alpha=0.9999):
        super().__init__()
        self.pool = pool
        self.num_blocks = num_blocks
        self.num_layers = num_layers

        self.enc = torch.nn.Sequential(torch.nn.Linear(in_channels, hidden_dim), torch.nn.ReLU())

        self.layers = torch.nn.ModuleList()
        self.poolings = torch.nn.ModuleList()
        for i in range(num_blocks):
            for j in range(num_layers):
                if conv_type == 'gin':
                    mlp = torch.nn.Sequential(
                        torch.nn.Linear(in_channels if i == 0 and j == -1 else hidden_dim, hidden_dim), torch.nn.ReLU(),
                        torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU())
                    self.layers.append(GINConv(mlp))#GCNConv(hidden_dim, hidden_dim))#
                elif conv_type == 'gcn':
                    self.layers.append(GCNConv(hidden_dim, hidden_dim))
                elif conv_type == 'gat':
                    self.layers.append(GATConv(hidden_dim, hidden_dim))

            if i < num_blocks - 1:
                if self.pool == 'xp':
                    mlp5 = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(),
                                               torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU())
                    mlp6 = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(),
                                               torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU())
                    self.poolings.append(
                        XPooling(in_channels=hidden_dim, mlp1=mlp5, mlp2=mlp6, alpha=alpha, merge=merge))
                elif self.pool == 'edge_pool':
                    self.poolings.append(EdgePooling(in_channels=hidden_dim))
                elif self.pool == 'topk':
                    self.poolings.append(TopKPooling(in_channels, min_score=0.2))
                elif self.pool == 'sag':
                    self.poolings.append(pyg.nn.SAGPooling(hidden_dim, ratio=0.8, min_score=0.2))
                elif self.pool == 'asa':
                    self.poolings.append(pyg.nn.ASAPooling(hidden_dim, ratio=0.8))
                elif self.pool == 'diff_pool':
                    self.poolings.append(DiffPool(hidden_dim, hidden_dim))

        self.dec = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(),
                                       torch.nn.Linear(hidden_dim, out_channels))

        self.global_pooling = global_add_pool

    def forward(self, data):
        edge_idx, batch = data.edge_index, data.batch
        x = self.enc(pyg.utils.degree(edge_idx[0], batch.shape[0]).unsqueeze(dim=-1))
        for i in range(self.num_blocks):
            for j in range(self.num_layers):
                x = self.layers[i * self.num_layers + j](x, edge_idx)
            if i < self.num_blocks - 1:
                if self.pool in ['xp', 'edge_pool', 'diff_pool']:
                    x, edge_idx, batch, _ = self.poolings[i](x, edge_idx, batch)
                elif self.pool in ['topk', 'sag']:
                    x, edge_idx, _, batch, _, _ = self.poolings[i](x, edge_idx, batch=batch)
                elif self.pool == 'asa':
                    x, edge_idx, _, batch, _ = self.poolings[i](x, edge_idx, batch=batch)

        x = self.global_pooling(x, batch)
        x = self.dec(x)

        return x
