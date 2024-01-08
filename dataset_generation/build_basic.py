import numpy as np
import networkx as nx
import utils
from tqdm import tqdm
import random
random.seed(2022)

file = 'Kac/processed/10a_nxwl_diff.npy'
g6_diff_list = np.load(file, allow_pickle=True)

g6_basic_list = []

for g6_list in g6_diff_list:
    if nx.is_regular(nx.from_graph6_bytes(g6_list[0].encode())):
        continue
    for id in range(0, len(g6_list) - 1, 2):
        g6_basic_list.append((g6_list[id], g6_list[id+1]))

g6_basic_list_random_selection = random.sample(g6_basic_list, 60)
random.shuffle(g6_basic_list_random_selection)

graph_diff = []
for g6 in g6_basic_list_random_selection:
    graph_diff.append(g6[0])
    graph_diff.append(g6[1])


# print(g6_basic_list_random_selection)
np.save('v3/raw/basic', graph_diff)
