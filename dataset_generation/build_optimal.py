import networkx as nx
import numpy as np
# import utils
from tqdm import tqdm
import random

random.seed(2022)

pyg_file_list = [f"optimal/graph_7_from_{i}_to_{i+75}.npy" for i in range(0, 751, 75)]
pyg_file_list.append("optimal/graph_6.npy")
wl_file_list = [
    f"optimal/wl_level_list_7_from_{i}_to_{i+75}.npy" for i in range(0, 751, 75)
]
wl_file_list.append("optimal/wl_level_list_6.npy")

wl_num_dict = {1: 60, 2: 20, 3: 20}
g6_tuple_each_wl_list = [[], [], []]

for i in tqdm(range(len(pyg_file_list))):
    pyg_list = np.load(pyg_file_list[i], allow_pickle=True)
    wl_list = np.load(wl_file_list[i], allow_pickle=True)
    for j in range(len(wl_list)):
        wl_level = wl_list[j]
        pyg_graph_1 = pyg_list[j * 2]
        pyg_graph_2 = pyg_list[j * 2 + 1]

        edge_tensor_1 = pyg_graph_1[0][1]
        edge_tensor_2 = pyg_graph_2[0][1]
        edges_1 = [
            (x.item(), y.item()) for x, y in zip(edge_tensor_1[0], edge_tensor_1[1])
        ]
        edges_2 = [
            (x.item(), y.item()) for x, y in zip(edge_tensor_2[0], edge_tensor_2[1])
        ]

        graph_1 = nx.Graph()
        graph_1.add_edges_from(edges_1)
        g6_1 = nx.to_graph6_bytes(graph_1, header=False).strip()
        graph_2 = nx.Graph()
        graph_2.add_edges_from(edges_2)
        g6_2 = nx.to_graph6_bytes(graph_2, header=False).strip()
        g6_tuple_each_wl_list[wl_level - 1].append((g6_1, g6_2))

g6_tuple_list_all = []
g6_set_test_all = set()
for wl, wl_num in wl_num_dict.items():
    print(wl, wl_num)
    g6_set = set()
    g6_tuple_list = []
    for g6_tuple in g6_tuple_each_wl_list[wl - 1]:
        if g6_tuple[0] in g6_set or g6_tuple[1] in g6_set:
            print('yes?')
            continue
        g6_set.add(g6_tuple[0])
        g6_set.add(g6_tuple[1])
        g6_tuple_list.append(g6_tuple)
    g6_tuple_list_random_selection = random.sample(g6_tuple_list, wl_num)
    for g6_tuple in g6_tuple_list_random_selection:
        g6_tuple_list_all.append(g6_tuple)

# random.shuffle(g6_tuple_list_all)
np.save("v3/raw/optimal", g6_tuple_list_all)
print(len(g6_tuple_list_all))
