import numpy as np
import argparse
from tqdm import tqdm
import torch
from torch_geometric.data import InMemoryDataset
from utils import graph6_to_pyg
import random


class KacDataset(InMemoryDataset):
    def __init__(
        self,
        root="Kac",
        device="cpu",
        raw_name=None,
        processed_name=None,
        shuffle=False,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.root = root
        self.device = device
        self.raw_name = raw_name
        self.processed_name = processed_name
        self.shuffle = shuffle
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[1])

    @property
    def raw_file_names(self):
        return [self.raw_name]

    @property
    def processed_file_names(self):
        return [f"{self.processed_name}_diff.npy", f"{self.processed_name}.pt"]

    def process(self):
        hash_dict = dict()
        hash_set = set()
        hash_set_repeat = set()
        graph_to_hash = dict()
        graph_to_hash = np.load(self.raw_paths[0], allow_pickle=True).item()

        for graph, graph_hash in tqdm(graph_to_hash.items()):
            if graph_hash not in hash_set:
                hash_set.add(graph_hash)
            else:
                hash_set_repeat.add(graph_hash)

        graph_list = list()
        cnt = 0
        for graph, graph_hash in tqdm(graph_to_hash.items()):
            if graph_hash not in hash_set_repeat:
                continue
            if graph_hash not in hash_dict:
                hash_dict[graph_hash] = cnt
                graph_list.append([graph])
                cnt += 1
            else:
                id = hash_dict[graph_hash]
                graph_list[id].append(graph)
        graph_list = np.array(graph_list, dtype=object)
        np.save(self.processed_paths[0], graph_list)

        pyg_list = []
        for x in tqdm(graph_list):
            for id_1 in range(len(x)):
                graph_1 = x[id_1]
                for id_2 in range(id_1 + 1, len(x)):
                    graph_2 = x[id_2]
                    pyg_graph_1 = graph6_to_pyg(graph_1)
                    pyg_graph_2 = graph6_to_pyg(graph_2)
                    pyg_graph_1.x = torch.zeros(size=(pyg_graph_1.num_nodes, 1))
                    pyg_graph_2.x = torch.zeros(size=(pyg_graph_2.num_nodes, 1))
                    pyg_list.append(pyg_graph_1)
                    pyg_list.append(pyg_graph_2)

        if self.shuffle:
            random.shuffle(pyg_list)
        data, slices = self.collate(pyg_list)
        torch.save((data, slices), self.processed_paths[1])


parser = argparse.ArgumentParser(description="Create graph-8c dataset.")
parser.add_argument("--ac", type=str, default="a")
parser.add_argument("--node", type=int, default="8")
parser.add_argument("--wl", type=int, default="2")
parser.add_argument("--shuffle", type=bool, default=False)
parser.add_argument("--mode", type=str, default="none")
args = parser.parse_args()


def main():
    if args.mode == "none":
        raw_name = f"{args.node}{args.ac}_{args.wl}wl.npy"
        processed_name = f"{args.node}{args.ac}"
    else:
        raw_name = f"{args.node}{args.ac}_{args.wl}wl_mode_{args.mode}.npy"
        processed_name = f"{args.node}{args.ac}_mode_{args.mode}"

    print(raw_name)
    print(processed_name)
    dataset = KacDataset(
        raw_name=raw_name, processed_name=processed_name, shuffle=args.shuffle
    )

    print(dataset)
    print(dataset[0])


if __name__ == "__main__":
    main()
