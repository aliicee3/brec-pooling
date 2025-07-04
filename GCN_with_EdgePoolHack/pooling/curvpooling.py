from collections import defaultdict
import torch
import torch_scatter
import networkx as nx
import torch_geometric as pyg


def most_frequent(List):
    if List != []:
        return List.count(max(set(List), key = List.count))
    else:
        return 0


def balancedFormanCurvature(edges):
    if edges is None or len(edges)==0:
        return []
    num_nodes = max([x for edge in edges for x in edge])+1
    candidates = [[] for i in range(num_nodes)]
    for edge in edges:
        candidates[edge[0]].append(edge[1])
        candidates[edge[1]].append(edge[0]) 
    degrees = [len(candidates[i]) for i in range(num_nodes)]

    triangles = []
    # print("Calculated Degrees and Candidates")
    # print(degrees)
    # print(candidates)
    # print(edges)

    quads_i_temp = []
    quads_j_temp = []
    quads_i_temp_set = set()
    quads_j_temp_set = set()
    quads_i = []
    quads_j = []
    quads_i_set = []
    quads_j_set = []
    # For every Edge
    for edge in edges:
        # Triangles = Intersection of Candidates
        triangles.append(list(set(candidates[edge[0]]).intersection(set(candidates[edge[1]]))))
        # Verbunden mit i
        for k in candidates[edge[0]]:
            # Nicht verbunden mit j
            if k not in candidates[edge[1]] and k not in edge:
                # Verbunden mit Nachbar omega von j
                for omega in set(candidates[k]).intersection(candidates[edge[1]]):
                    # Omega nicht verbunden mit i
                    if omega not in candidates[edge[0]] and omega not in edge:
                        quads_i_temp.append((k,omega))
                        quads_i_temp_set.add(k)
        quads_i.append(quads_i_temp)
        quads_i_set.append(frozenset(quads_i_temp_set))
        quads_i_temp = []
        quads_i_temp_set = set()

        # Verbunden mit j
        for k in candidates[edge[1]]:
            # Nicht verbundne mit i
            if k not in candidates[edge[0]] and k not in edge:
                # Verbunden mit Nachbar omega von i
                for omega in set(candidates[k]).intersection(candidates[edge[0]]):
                    # Omega nicht verbunden mit j
                    if omega not in candidates[edge[1]] and omega not in edge:
                        quads_j_temp.append((k,omega))
                        quads_j_temp_set.add(k)
        quads_j.append(quads_j_temp)
        quads_j_set.append(frozenset(quads_j_temp_set))
        quads_j_temp = []
        quads_j_temp_set = set()

    # print("Calculated Triangles and Quads")
    # print(triangles)
    # print(quads_i)
    # print(quads_j)

    num_quads = []
    # Calculate Maximum Number of Quads with a shared Node
    for i in range(len(quads_i)):
        counts_i = [x for quad in quads_i[i] for x in quad]
        counts_j = [x for quad in quads_j[i] for x in quad]
        num_quads.append(max(most_frequent(counts_i),most_frequent(counts_j)))

    # print("Calculated Number of Quads")
    # print(num_quads)

    # Calculate Balanced Forman Curvature
    bfc = []
    for edge in edges:
        edge_index = edges.index(edge)
        min_degree = min(degrees[edge[0]],degrees[edge[1]])
        max_degree = max(degrees[edge[0]],degrees[edge[1]])
        result = 0.0
        if min_degree == 1:
            result = 0.0
        elif len(quads_i[edge_index]) == 0:
            result = 2/degrees[edge[0]] + 2/degrees[edge[1]] - 2 + 2 * len(triangles[edge_index])/max_degree + len(triangles[edge_index])/min_degree
        else:
            result = 2/degrees[edge[0]] + 2/degrees[edge[1]] - 2 + 2 * len(triangles[edge_index])/max_degree + len(triangles[edge_index])/min_degree + num_quads[edge_index]**(-1)/max_degree * (len(quads_i_set[edge_index])+len(quads_j_set[edge_index]))
        bfc.append(result)
    return bfc



#merge function to  merge all sublist having common elements.
def merge_common(lists):
    neigh = defaultdict(set)
    visited = set()
    for each in lists:
        for item in each:
            neigh[item].update(each)
    def comp(node, neigh = neigh, visited = visited, vis = visited.add):
        nodes = set([node])
        next_node = nodes.pop
        while nodes:
            node = next_node()
            vis(node)
            nodes |= neigh[node] - visited
            yield node
    for node in neigh:
        if node not in visited:
            yield sorted(comp(node))


def precalc_dense_curv_mappings(adj, avg_pool=False, threshold=None, merge=True):
    # Simulate single Batch, if no Batches by adding a new dimension of size 1
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj

    batch_num, num_nodes, _ = adj.size()

    adjacency_matrices = []
    curvature_mappings = []
    for batch in range(batch_num):
        edge_index, edge_attributes = pyg.utils.dense_to_sparse(adj[batch])
        edge_index = pyg.utils.remove_self_loops(edge_index)[0]
        edges = [(a.item(),b.item()) for a,b in zip(edge_index[0],edge_index[1])]
        #num_nodes = max(max(edge_index[0]),max(edge_index[1])+1)
        for a,b in edges:
            if (a,b) in edges:
                edges.remove((b,a))

        bfc = balancedFormanCurvature(edges)

        # Create Curvature Groups for all edges fullfilling the requirement
        if threshold == None:
            if len(bfc) == 0:
                threshold = 0
            else:
                threshold = sum(bfc) / len(bfc)

        curv_groups = [set(a for a in edge) for edge,curv in zip(edges,bfc) if curv>threshold]

        # Merge Groups with intersections
        if merge:
            curv_groups = list(merge_common(curv_groups))

        # Keep Nodes that arent part of any Curvature Groups
        for node in range(adj.shape[1]): # CHANGED 0 -> 1
            # Node has an Edge or is the last remaining node
            if node in [a for edge in edges for a in edge] or len(curv_groups) == 0:
                if node not in [curv_node for curv_group in curv_groups for curv_node in curv_group]:
                    curv_groups.append([node])

        # print(f"Curv groups: {curv_groups}")

        # Construct curvature_mapping
        if avg_pool:
            curvature_mapping = torch.tensor([[1./len(curv_group) if node in curv_group else 0. for curv_group in curv_groups] for node in range(num_nodes)])
        else:
            curvature_mapping = torch.tensor([[1. if node in curv_group else 0. for curv_group in curv_groups] for node in range(num_nodes)])

        # print(curvature_mapping)
        # print(curvature_mapping.shape)

        # Komplexere Adjazenzmatrix
        adj = adj.type(torch.FloatTensor)
        curvature_mapping = torch.nn.functional.pad(curvature_mapping,(0,0,0,adj.shape[1]-curvature_mapping.shape[0]),"constant",0)
        curvature_mapping = curvature_mapping

        adjacency_matrix = torch.matmul(torch.matmul(curvature_mapping.transpose(0, 1), adj[batch]), curvature_mapping).type(torch.BoolTensor).type(torch.LongTensor)
        curvature_mappings.append(curvature_mapping)
        adjacency_matrices.append(adjacency_matrix)

        max_len = adj.shape[1]
        adjacency_matrix = torch.nn.functional.pad(adjacency_matrix,(0,max_len-adjacency_matrix.shape[1],0,max_len-adjacency_matrix.shape[1]),"constant",0)

        # print(adj.shape)
        # print(adjacency_matrix.shape)
        # print(torch.eq(adj[batch],adjacency_matrix))

        # print(adjacency_matrix.shape)
        # print(adjacency_matrix)

    # Pad with Zeroes
    max_len_x = max([element.shape[0] for element in curvature_mappings]) # adj.shape[1]
    max_len_y = max([element.shape[1] for element in curvature_mappings]) # adj.shape[1]
    padded = []
    for element in curvature_mappings:
        new_element = torch.nn.functional.pad(element,(0,max_len_y-element.shape[1],0,max_len_x-element.shape[0]),"constant",0)
        padded.append(new_element)
    curvature_mappings = torch.tensor([element.tolist() for element in padded], device=adj.device)
    max_len =  max([element.shape[0] for element in adjacency_matrices]) # adj.shape[1]
    padded = []
    for element in adjacency_matrices:
        new_element = torch.nn.functional.pad(element,(0,max_len-len(element),0,max_len-len(element)),"constant",0)
        padded.append(new_element)
    adjacency_matrices = torch.tensor([element.tolist() for element in padded], device=adj.device)

    # print(adj.shape)
    # print(adjacency_matrices.shape)
    # print(torch.eq(adj,adjacency_matrices))

    # print(curvature_mappings.shape)
    # print(curvature_mappings)
    # print(adjacency_matrices.shape)
    # print(adjacency_matrices)
    return adjacency_matrices, curvature_mappings

def dense_curv_pool(x, s, mask=None, normalize=True):
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

class CurvPooling(torch.nn.Module):
    def __init__(self):
       super().__init__()
       self.dense_to_sparse = DenseToSparse()

    def forward(self, x, edge_index, batch):
        A = pyg.utils.to_dense_adj(edge_index,batch)
        A_new, S = precalc_dense_curv_mappings(A)
        if A_new is None:
            return x, edge_index, batch, None
        x_dense, _ = pyg.utils.to_dense_batch(x, batch)
        x_new = dense_curv_pool(x_dense, S)
        x_new, edge_index_new, _, batch_new = self.dense_to_sparse(x_new, A_new)
        return x_new, edge_index_new, batch_new, None