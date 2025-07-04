# CLIQUE POOL Ansatz
import torch
import torch_scatter
import networkx as nx
import torch_geometric as pyg

def precalc_dense_mappings(adj):
    # Simulate single Batch, if no Batches by adding a new dimension of size 1
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj

    batch_num,N,_ = adj.size()

    adjacency_matrices = []
    clique_mappings = []
    for batch in range(batch_num):
        edge_indices, edge_attributes = pyg.utils.dense_to_sparse(adj[batch])
        temp_adj = pyg.utils.to_scipy_sparse_matrix(edge_indices)
        graph = nx.from_scipy_sparse_array(temp_adj) #nx.Graph()

        # Simple Clique Mapping
        # cliques = list(nx.clique.find_cliques(graph))
        # if avg_pool:
        #     clique_mapping = torch.tensor([[1./len(clique) if node in clique else 0. for clique in cliques] for node in graph.nodes()])
        # else:
        #     clique_mapping = torch.tensor([[1. if node in clique else 0. for clique in cliques] for node in graph.nodes()])
        # clique_mappings.append(clique_mapping)
        # print(clique_mapping.shape)
        # print(clique_mapping)

        # Komplexeres Clique Mapping
        cliques = list(nx.clique.find_cliques(graph))
        nodes = list(graph.nodes())
        changed = True
        #if batch == 1 and len(cliques) > 100:
        #   print(len(cliques),len(nodes))
        if len(cliques)> 100:
            print(len(cliques),len(nodes))
            adjs = torch.ones((batch_num,1,1),device=adj.device)
            s = torch.ones((batch_num,N,1),device=adj.device) 
            return adjs, s
        while changed:
            max_cliques = []
            for node in nodes:
                max_clique_size = 0
                max_clique = []
                for clique in cliques:
                    if node in clique:
                        if max_clique_size < len(clique):
                            max_clique_size = len(clique)
                            max_clique = [cliques.index(clique)]
                        elif max_clique_size == len(clique):
                            max_clique.append(cliques.index(clique))
                max_cliques.append(max_clique)
            #print(max_cliques)
            changed = False
            # Entferne den Knoten aus allen nicht größten Cliquen
            for clique in cliques:
                for node in nodes:
                    if node in clique and cliques.index(clique) not in max_cliques[nodes.index(node)]:
                        changed = True
                        clique.remove(node)
        #print(cliques)

        # Entferne Duplikate und leere Cliquen
        cliques = [element for element in cliques if element != []]
        unique_cliques = []
        for clique in cliques:
            if clique not in unique_cliques:
                unique_cliques.append(clique)
        cliques = unique_cliques
        clique_mapping = torch.tensor([[1. if node in clique else 0. for clique in cliques] for node in graph.nodes()])
        #print(clique_mapping)
        #print(clique_mapping.shape)

        # Komplexere Adjazenzmatrix
        adj = adj.type(torch.FloatTensor)#.to(device)
        clique_mapping = torch.nn.functional.pad(clique_mapping,(0,0,0,adj.shape[1]-clique_mapping.shape[0]),"constant",0)
        clique_mapping = clique_mapping#.to(device)

        adjacency_matrix = torch.matmul(torch.matmul(clique_mapping.transpose(0, 1), adj[batch]), clique_mapping).type(torch.BoolTensor).type(torch.LongTensor)
        clique_mappings.append(clique_mapping)
        adjacency_matrices.append(adjacency_matrix)
        #print(adjacency_matrix.shape)
        #print(adjacency_matrix)

    # Pad with Zeroes
    max_len_x = max([element.shape[0] for element in clique_mappings])
    max_len_y = max([element.shape[1] for element in clique_mappings])
    padded = []
    for element in clique_mappings:
        new_element = torch.nn.functional.pad(element,(0,max_len_y-element.shape[1],0,max_len_x-element.shape[0]),"constant",0)
        padded.append(new_element)
    clique_mappings = torch.tensor([element.tolist() for element in padded], device=adj.device)
    max_len = max([element.shape[0] for element in adjacency_matrices])
    padded = []
    for element in adjacency_matrices:
        new_element = torch.nn.functional.pad(element,(0,max_len-len(element),0,max_len-len(element)),"constant",0)
        padded.append(new_element)
    adjacency_matrices = torch.tensor([element.tolist() for element in padded], device=adj.device)

    # print(clique_mappings.shape)
    # print(clique_mappings)
    # print(adjacency_matrices.shape)
    # print(adjacency_matrices)
    return adjacency_matrices, clique_mappings

def dense_clique_pool(x, s, mask=None, normalize=True):
    # Simulate single Batch, if no Batches by adding a new dimension of size 1
    x = x.unsqueeze(0) if x.dim() == 2 else x
    s = s.unsqueeze(0) if s.dim() == 2 else s

    # Get batch_size and num_nodes
    batch_size, num_nodes, _ = x.size()

    # Apply mask(graph-node-mapping?) to x and s
    if mask is not None:
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask
    # calculate aggregated features + aggregated adjacency matrix
    # Pooling Style = SUM,AVG,MAX
    out = torch.matmul(s.transpose(1, 2), x)

    return out 

class DenseToSparse(torch.nn.Module):
    """Convert from adj to edge_list while allowing gradients
    to flow through adj"""

    def forward(self, x, adj):
        B = x.shape[0]
        N = x.shape[1]
        offset, row, col = torch.nonzero(adj > 0).t()
        row += offset * N
        col += offset * N
        edge_index = torch.stack([row, col], dim=0).long()
        x = x.view(B * N, x.shape[-1])
        batch_idx = (
            torch.arange(0, B, device=x.device).view(-1, 1).repeat(1, N).view(-1)
        )

        return x, edge_index, None, batch_idx

class CliquePooling(torch.nn.Module):
    def __init__(self):
       super().__init__()
       self.dense_to_sparse = DenseToSparse()

    def forward(self, x, edge_index, batch):
        #print(x.shape, edge_index.shape, batch.shape)        
        A = pyg.utils.to_dense_adj(edge_index,batch)
        #print(A.shape)
        A_new, S = precalc_dense_mappings(A)
        if A_new is None:
            return x, edge_index, batch, None
        #print(A_new.shape, S.shape)
        x_dense, _ = pyg.utils.to_dense_batch(x, batch)
        #print(x_dense.shape)
        x_new = dense_clique_pool(x_dense, S)
        #print(x_new.shape)
        #edge_index_new, _ = pyg.utils.dense_to_sparse(A_new)
        #print(edge_index_new.shape)
        x_new, edge_index_new, _, batch_new = self.dense_to_sparse(x_new, A_new)
        return x_new, edge_index_new, batch_new, None
        