import utils
import networkx as nx
import numpy as np
import random

random.seed(2022)
mode_1wl_graph_file_list = [
    "Kac/processed/10a_mode_n1_diff.npy",
    "Kac/processed/10a_mode_s3_diff.npy",
    "Kac/processed/10a_mode_s4_diff.npy",
]
graph_n1_file = mode_1wl_graph_file_list[0]
graph_s3_file = mode_1wl_graph_file_list[1]
graph_s4_file = mode_1wl_graph_file_list[2]
graph_n1_list = np.load(graph_n1_file, allow_pickle=True)
graph_s3_list = np.load(graph_s3_file, allow_pickle=True)
graph_s4_list = np.load(graph_s4_file, allow_pickle=True)

list_n1 = []
for x in graph_n1_list:
    for id_1 in range(0, len(x) - 1, 2):
        graph_1 = x[id_1]
        graph_2 = x[id_1 + 1]
        list_n1.append(tuple((graph_1, graph_2)))

list_s3 = []
for x in graph_s3_list:
    for id_1 in range(0, len(x) - 1, 2):
        graph_1 = x[id_1]
        graph_2 = x[id_1 + 1]
        list_s3.append(tuple((graph_1, graph_2)))

list_s4 = []
for x in graph_s4_list:
    for id_1 in range(0, len(x) - 1, 2):
        graph_1 = x[id_1]
        graph_2 = x[id_1 + 1]
        list_s4.append(tuple((graph_1, graph_2)))

list_s3_select = random.sample(list_s3, 60)
list_s4_select = random.sample(list_s4, 10)
list_n1_select = random.sample(list_n1, 20)

# cnt = 0
# for g6_tuple in list_s3_select:
#     g0 = nx.from_graph6_bytes(g6_tuple[0].encode())
#     g1 = nx.from_graph6_bytes(g6_tuple[1].encode())
#     if utils.WL_1_hash(g0, mode='n1') == utils.WL_1_hash(g1, mode='n1'):
#         cnt += 1
# print(cnt)


graph_diff = []
graph_diff.extend(list_s3_select)
graph_diff.extend(list_n1_select)
graph_diff.extend(list_s4_select)
print(len(graph_diff))

g6_set = set()
for g6_tuple in list_s3_select:
    g6_set.add(g6_tuple[0])
    g6_set.add(g6_tuple[1])

for g6_tuple in list_s4_select:
    g6_set.add(g6_tuple[0])
    g6_set.add(g6_tuple[1])

for g6_tuple in list_n1_select:
    g6_set.add(g6_tuple[0])
    g6_set.add(g6_tuple[1])

print(len(g6_set))

for i in range(5, 10):
    g6_1 = nx.to_graph6_bytes(utils.cll(i), header=False).strip()
    g6_2 = nx.to_graph6_bytes(utils.c2l(i), header=False).strip()
    graph_diff.append((g6_1, g6_2))

regular_file = "v3/raw/regular.npy"
regular = list(np.load(regular_file, allow_pickle=True))
regular_selection = random.sample(regular, 5)
for g6_tuple in regular_selection:
    graph_tuple = (nx.from_graph6_bytes(g6_tuple[0]), nx.from_graph6_bytes(g6_tuple[1]))
    u = np.max(list(graph_tuple[0].nodes())) + 1
    graph_tuple[0].add_edges_from([(i, u) for i in range(u)])
    graph_tuple[1].add_edges_from([(i, u) for i in range(u)])
    g6_tuple_revised = (
        nx.to_graph6_bytes(graph_tuple[0], header=False).strip(),
        nx.to_graph6_bytes(graph_tuple[1], header=False).strip(),
    )
    print(g6_tuple_revised[0])
    print(g6_tuple_revised[1])
    graph_diff.append(g6_tuple_revised)

print(len(graph_diff))
np.save("v3/raw/special.npy", graph_diff)
