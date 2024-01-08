import numpy as np
import networkx as nx
import random
import utils

random.seed(2022)
file = 'g6/srg_63_30_13_15_4vtx.g6'
with open(file, "r") as f:
    g6_list = [x.strip() for x in f.readlines()]

num = 40
graph_diff = []
g6_list_random_selection = random.sample(list(g6_list), num)
random.shuffle(g6_list_random_selection)

np.save('v3/raw/4vtx', g6_list_random_selection)
# for g6 in g6_list_random_selection:
#     graph_diff.append(nx.from_graph6_bytes(g6.encode()))

# # print(len(graph_diff))
# # # for g in graph_diff:
# # #     print(utils.FWL_hashlib(g, 2))
# # #     print(utils.FWL_hashlib(g, 3))
# print(graph_diff)
# graph_diff = np.asanyarray(graph_diff, dtype=object)
# print(graph_diff)
# np.save('v3/raw/4vtx', graph_diff)
