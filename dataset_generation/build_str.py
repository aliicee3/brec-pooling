import networkx as nx
import numpy as np
import random
import utils

random.seed(2022)

graph_diff = []
file_list = [
    'sr251256.g6',
    'sr261034.g6',
    'sr281264.g6',
    'sr291467.g6',
    'sr351899.g6'
]

graph_diff.append(nx.to_graph6_bytes(utils.rooks_graph(4, 4), header=False).strip())
graph_diff.append(nx.to_graph6_bytes(utils.shrikhande_graph(), header=False).strip())

for file in file_list:
    with open(f'g6/{file}', 'r') as f:
        g6_list = [x.strip() for x in f.readlines()]
    if len(g6_list) > 50:
        g6_list_random_selection = random.sample(g6_list, 30)
    else:
        g6_list_random_selection = random.sample(g6_list, len(g6_list) // 2 * 2)
    # random.shuffle(g6_list_random_selection)
    graph_diff.extend(g6_list_random_selection)

print(len(graph_diff))
np.save('v3/raw/str', graph_diff)
