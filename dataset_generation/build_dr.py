import numpy as np
import random
import networkx as nx
# import utils

random.seed(2022)
np.random.seed(2022)

file_list = [
    'dual_ah22a',
    'hada_1573',
    'hada_1994',
    'hada_2593',
    'hada_23115',
    'incidence_1662',
    'incidence_16106',
]


graph_diff = []
for f in file_list:
    g6_list = np.load(f'dr/{f}.npy', allow_pickle=True)
    if len(g6_list) < 10:
        for id in range(0, len(g6_list) - 1, 2):
            graph_diff.append(g6_list[id])
            graph_diff.append(g6_list[id + 1])
    elif f == 'hada_2593':
        num = 8
        g6_list_random_selection = random.sample(list(g6_list), num)
        for g6 in g6_list_random_selection:
            graph_diff.append(g6)
    elif f == 'hada_23115':
        num = 16
        g6_list_random_selection = random.sample(list(g6_list), num)
        for g6 in g6_list_random_selection:
            graph_diff.append(g6)
    else:
        raise NotImplementedError(f'{f} should have less than 5 pairs')

graph_diff = [nx.to_graph6_bytes(nx.from_graph6_bytes(g6), header=False).strip() for g6 in graph_diff]

print(len(graph_diff))
# graph_diff = np.asanyarray(graph_diff, dtype=object)
np.save('v3/raw/dr', graph_diff)
