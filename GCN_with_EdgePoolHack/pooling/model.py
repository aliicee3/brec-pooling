import torch
from torch.nn import Linear, Parameter, Module
from torch_geometric.nn import GCNConv, GINConv, GATConv
from torch_geometric.nn.pool import global_add_pool, TopKPooling, EdgePooling
from torch_geometric.utils import add_self_loops, degree
from pooling.edge_pool_hack import EdgePoolingHack
import torch_geometric as pyg

class GINandPool(torch.nn.Module):
    
    POOLING_OPTIONS = ['edge_pool', 'edge_pool_base', 'topk']
    def __init__(self, in_channels, hidden_dim, out_channels, pool='topk', num_blocks=3, num_layers=4, conv_type='gin'):
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
                if self.pool == 'edge_pool':
                    mlp5 = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(),
                                               torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU())
                    mlp6 = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(),
                                               torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU())
                    self.poolings.append(
                        EdgePoolingHack(in_channels=hidden_dim, mlp1=mlp5, mlp2=mlp6, deterministic=False))
                elif self.pool == 'edge_pool_base':
                    self.pooling = EdgePooling(in_channels=hidden_dim)                
                elif self.pool == 'topk':
                    self.poolings.append(TopKPooling(in_channels))

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
                if self.pool in ['edge_pool', 'edge_pool_base']:
                    x, edge_idx, batch, _ = self.poolings[i](x, edge_idx, batch)
                elif self.pool == 'topk':
                    x, edge_idx, _, batch, _, _ = self.poolings[i](x, edge_idx, batch=batch)

        x = self.global_pooling(x, batch)
        x = self.dec(x)

        return x
