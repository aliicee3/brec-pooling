import torch
from torch.nn import Linear, Parameter, Module
from torch_geometric.nn import GCNConv, GINConv 
from torch_geometric.nn.pool import global_add_pool, TopKPooling, EdgePooling
from torch_geometric.utils import add_self_loops, degree
from pooling.edge_pool_hack import EdgePoolingHack


class GINandPool(torch.nn.Module):

    POOLING_OPTIONS = ['edge_pool', 'edge_pool_base', 'topk']
    
    def __init__(self, in_channels, hidden_dim, out_channels, pool='topk'):
        super().__init__() 
        self.mlp1 = torch.nn.Sequential(torch.nn.Linear(in_channels, hidden_dim), torch.nn.ReLU(), torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU())
        self.mlp2 = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(), torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU())
        self.mpnn1 = GINConv(self.mlp1)
        self.mpnn2 = GINConv(self.mlp2)

        self.mlp3 = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(), torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU())
        self.mlp4 = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(), torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU())        
        self.mpnn3 = GINConv(self.mlp3)
        self.mpnn4 = GINConv(self.mlp4)

        self.pool = pool
        if self.pool == 'edge_pool':
            self.mlp5 = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(), torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU())
            self.mlp6 = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(), torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU())
            self.pooling = EdgePoolingHack(in_channels=hidden_dim, mlp1=self.mlp5, mlp2=self.mlp6)
        elif self.pool == 'edge_pool_base':
            self.pooling = EdgePooling(in_channels=hidden_dim)
        elif self.pool == 'topk':
            self.pooling = TopKPooling(in_channels) 
        
        self.dec = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(), torch.nn.Linear(hidden_dim, out_channels))
        self.global_pooling = global_add_pool


    def forward(self, data):
        edge_idx, batch = data.edge_index, data.batch 
        x = torch.ones([batch.shape[0], 1],device=edge_idx.device)
        x = self.mpnn1(x, edge_idx)
        x = self.mpnn2(x, edge_idx)
        if self.pool in ['edge_pool', 'edge_pool_base']:
            x, edge_idx, batch, _ = self.pooling(x, edge_idx, batch)
        elif self.pool == 'topk':
            x, edge_idx, _, batch, _, _ = self.pooling(x, edge_idx, batch=batch)
        x = self.mpnn3(x, edge_idx)
        x = self.mpnn4(x, edge_idx)
        
        x = self.global_pooling(x, batch)
        x = self.dec(x)

        return x
