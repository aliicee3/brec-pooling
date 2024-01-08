from email import header
import networkx as nx
import numpy as np
import os
import random

random.seed(2022)
path = "asc/"

file_list = os.listdir(path)
g6_tuple_list = []

for file in file_list:
    with open(f"{path}{file}", "r") as f:
        file_lines = [x.strip() for x in f.readlines() if x != "\n"]
    n = int(file.split("_")[0])
    print(n)
    cnt = 0
    graph_list = []
    for (i, line) in enumerate(file_lines):
        if line[0] == "G":
            cnt += 1
            G = nx.Graph()
            G.add_nodes_from(list(range(1, n + 1)))
            for j in range(i + 1, i + n + 1):
                (u, v_list) = file_lines[j].split(" : ")
                v_list = v_list.split(" ")
                for v in v_list:
                    G.add_edge(int(u), int(v))
            graph_list.append(nx.to_graph6_bytes(G, header=False).strip())

    for id in range(0, len(graph_list) - 1, 2):
        g6_tuple_list.append((graph_list[id], graph_list[id + 1]))

g6_tuple_list_random_selection = random.sample(g6_tuple_list, 50)
# random.shuffle(g6_tuple_list_random_selection)
for g6_tuple in g6_tuple_list_random_selection:
    print(nx.from_graph6_bytes(g6_tuple[0]).number_of_nodes())

np.save("v3/raw/regular", g6_tuple_list_random_selection)
