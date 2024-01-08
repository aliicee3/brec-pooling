import random
import numpy as np
import networkx as nx
import argparse
from tqdm import tqdm
import torch
from torch_geometric.data import InMemoryDataset
from collections import Counter
from utils import (
    fwl_hash_tqdm,
    test_optimal_graph_degree,
    construct_optimal_graph,
    # save_graph,
    transform_to_geometric,
)




class WLDataset(InMemoryDataset):
    def __init__(
        self,
        root="data",
        device="cpu",
        graph_list=None,
        max_node_num=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.root = root
        self.device = device
        self.graph_list = graph_list
        self.max_node_num = max_node_num
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return

    @property
    def processed_file_names(self):
        return [f"optimal_wl_dataset_{self.max_node_num}.pt"]

    def process(self):
        data, slices = self.collate(self.graph_list)
        torch.save((data, slices), self.processed_paths[0])


random.seed(2022)
np.random.seed(2022)
parser = argparse.ArgumentParser(
    description="Create a dataset with k-wl expressive ability."
)
parser.add_argument("--type", type=str, default="optimal")
# parser.add_argument("--backbone", type=str, default="er")
parser.add_argument("--node_num", type=int, default=7)
parser.add_argument("--size", type=int, default=75)
parser.add_argument("--pos", type=int, default=0)
# parser.add_argument("--max_node_num", type=int, default=6)
# parser.add_argument("--start_node_num", type=int, default=3)

# parser.add_argument("--density", type=float, default=0.5)
# parser.add_argument("--num", type=int, default=1)
# parser.add_argument("--wl_level", type=int, default=1)
args = parser.parse_args()


pyg_iso_list = []
wl_level_list = []
id_list = []
start_pos = args.pos * args.size
end_pos = (args.pos + 1) * args.size
print("starting---")

print(f"backbone_node_num:{args.node_num}")
print(f"from {start_pos} to {end_pos}: ")
with open(f"g6/graph{args.node_num}c.g6", "r") as f:
    graphkac = [x.strip() for x in f.readlines()]
for id in tqdm(range(start_pos, end_pos)):
    if id >= len(graphkac):
        break
    x = graphkac[id]
    G = nx.from_graph6_bytes(x.encode())
    if not test_optimal_graph_degree(G):
        continue
    (G_iso_1, G_iso_2) = construct_optimal_graph(G)
    fwl_level_indistinguish = 3
    for k in range(2, 4):
        if not fwl_hash_tqdm(G_iso_1, k, debug=False) == fwl_hash_tqdm(
            G_iso_2, k, debug=False
        ):
            fwl_level_indistinguish = k - 1
            break
    (pyg_iso_1, pyg_iso_2) = transform_to_geometric(G_iso_1, G_iso_2)
    pyg_iso_1.x = torch.zeros(size=(pyg_iso_1.num_nodes, 1))
    pyg_iso_2.x = torch.zeros(size=(pyg_iso_2.num_nodes, 1))
    pyg_iso_list.append(pyg_iso_1)
    pyg_iso_list.append(pyg_iso_2)
    wl_level_list.append(fwl_level_indistinguish)
    id_list.append(id)


counter = Counter(wl_level_list)
print(counter)
print(f'Saving--{args.node_num}_from_{start_pos}_to_{end_pos}')
pyg_iso_list = np.array(pyg_iso_list, dtype=object)
wl_level_list = np.array(wl_level_list, dtype=object)
id_list = np.array(id_list, dtype=object)
np.save(
    f"optimal/wl_level_list_{args.node_num}_from_{start_pos}_to_{end_pos}.npy",
    wl_level_list,
)
np.save(
    f"optimal/graph_{args.node_num}_from_{start_pos}_to_{end_pos}.npy", pyg_iso_list
)
np.save(f"optimal/id_list_{args.node_num}_from_{start_pos}_to_{end_pos}.npy", id_list)


def main():
    pass
    # dataset = WLDataset(graph_list=pyg_iso_list, max_node_num=args.max_node_num)
    # print(len(dataset))
    # print(dataset[0])


if __name__ == "__main__":
    main()
