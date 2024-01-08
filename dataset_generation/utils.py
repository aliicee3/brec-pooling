from itertools import product
from math import gcd
import networkx as nx
import matplotlib.pyplot as plt
import copy

# import torch
# from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx, to_networkx
from collections import Counter
from tqdm import tqdm
import numpy as np
import random
import hashlib
import pynauty
import re


def get_subset_even(nums):
    sub_sets = [[]]
    for x in nums:
        sub_sets.extend([item + [x] for item in sub_sets])
    sub_sets = list(filter(lambda x: len(x) % 2 == 0, sub_sets))
    return sub_sets


def generate_graph_X(node, neighbors):
    g = nx.Graph()
    for i in neighbors:
        g.add_node("a_" + str(node) + "_" + str(i))
        g.add_node("b_" + str(node) + "_" + str(i))
    # for i in range(1, n+1):
    #     g.add_node(str(node) +'_a'+str(i))
    #     g.add_node(str(node)+'_b'+str(i))
    # n = len(neighbors)
    M = get_subset_even(neighbors)
    for x in M:
        if len(x) == 0:
            node_name = str(node) + "_m_0"
            g.add_node(node_name)
            g.add_edges_from(
                [(node_name, "b_" + str(node) + "_" + str(i)) for i in neighbors]
            )
        else:
            node_name = str(node) + "_m"
            for id in x:
                node_name += "_" + str(id)
            g.add_node(node_name)
            s = 0
            for i in neighbors:
                if s == len(x):
                    g.add_edge(node_name, "b_" + str(node) + "_" + str(i))
                elif x[s] == i:
                    s += 1
                    g.add_edge(node_name, "a_" + str(node) + "_" + str(i))
                else:
                    g.add_edge(node_name, "b_" + str(node) + "_" + str(i))
    return g


def connect_XG(G):
    X = nx.Graph()
    for node in G.nodes:
        tmp = generate_graph_X(node=node, neighbors=list(G.neighbors(node)))
        X.add_edges_from(tmp.edges)
        # print(X.number_of_nodes())
    for node in G.nodes:
        tmp = generate_graph_X(node=node, neighbors=list(G.neighbors(node)))
        for node_tmp in tmp.nodes:
            uv_list = node_tmp.split("_")
            if uv_list[0] == "a" or uv_list[0] == "b":
                X.add_edge(node_tmp, uv_list[0] + "_" + uv_list[2] + "_" + uv_list[1])
    return X


def twist_optimal_graph(G, u, v):
    H = nx.Graph(G)
    H.remove_edge("a_" + str(u) + "_" + str(v), "a_" + str(v) + "_" + str(u))
    H.remove_edge("b_" + str(u) + "_" + str(v), "b_" + str(v) + "_" + str(u))
    H.add_edge("a_" + str(u) + "_" + str(v), "b_" + str(v) + "_" + str(u))
    H.add_edge("a_" + str(v) + "_" + str(u), "b_" + str(u) + "_" + str(v))
    return H


def test_optimal_graph_degree(G):
    min_degree = min(G.degree, key=lambda dic: dic[1])[1]
    if min_degree < 2:
        return False
    return True


def test_optimal_graph(G, wl_level):
    min_degree = min(G.degree, key=lambda dic: dic[1])[1]
    if min_degree < 2:
        return 0
    k = len(nx.minimum_node_cut(G))
    # H = nx.Graph(G)
    # n = H.number_of_nodes()
    # H.add_node("s")
    # H.add_node("t")
    # s_edge = [("s", i) for i in range(n)]
    # t_edge = [("t", i) for i in range(n)]
    # H.add_edges_from(s_edge)
    # H.add_edges_from(t_edge)
    # for edge in H.edges:
    #     H.add_edge(edge[0], edge[1], capacity=1)
    # k = nx.maximum_flow(H, _s="s", _t="t")[0]
    # return k == wl_level + 2
    print(f"{k-2} fwl_level indistinguishable")
    return k == wl_level + 2


def construct_optimal_graph(G):
    G1 = connect_XG(G)
    # print(G1.edges)
    for edge in G1.edges:
        if edge[0][0] == "a" and edge[1][0] == "a":
            tmp_list = edge[0].split("_")
            G2 = twist_optimal_graph(G1, tmp_list[1], tmp_list[2])
            return (G1, G2)


def save_graph(G, k):
    nx.draw(G)
    plt.savefig(f"pic/{k}_{G}.png")
    plt.close()
    return


def transform_to_geometric(G1, G2):
    pyg_graph1 = from_networkx(G1)
    pyg_graph2 = from_networkx(G2)
    pyg_graph1.y = 0
    pyg_graph2.y = 1
    return (pyg_graph1, pyg_graph2)


def get_cnt(node_list, node_num, node_to_id):
    result = 0
    for x in node_list:
        result *= node_num
        result += node_to_id[x]
    return result


def wl_hash_c(G, k, debug=False):
    # k should satisfy that any subgraph with k nodes is distinguishable with 1-WL
    node_list = [x for x in G.nodes]
    # print('node_list', node_list)
    n = G.number_of_nodes()
    node_to_id = dict()
    cnt = 0
    for i in range(n):
        node_to_id[node_list[i]] = i
    hash_wl_discret = dict()  # discretization of wl hash value
    node_vector_hash = []  # hash_value for each vector
    node_vector_hash_discret = []  # hash_value discretization for each vector
    node_vector_list = list(product(node_list, repeat=k))

    s = 0
    if debug:
        print("preprocessing---")
    for node_vector in node_vector_list:
        if debug:
            if s % 1000000 == 0:
                print(s)
        s += 1
        sub_G = G.subgraph(node_vector)
        hash_wl = nx.weisfeiler_lehman_graph_hash(sub_G)
        # hash_wl = sub_G.number_of_edges()
        if hash_wl not in hash_wl_discret:
            hash_wl_discret[hash_wl] = cnt
            cnt += 1
        node_vector_hash.append(hash_wl)
        node_vector_hash_discret.append(hash_wl_discret[hash_wl])

    epoch = 1
    while epoch:
        if debug:
            print(f"Epoch {epoch}:")
        epoch += 1
        hash_wl_discret = dict()
        node_vector_hash_nxt = []
        node_vector_hash_discret_nxt = []
        cnt = 0
        for id in range(len(node_vector_list)):
            if debug:
                if id % 1000000 == 0:
                    print(id)
            node_vector = node_vector_list[id]

            # hash_l_list is c^l_v, hash(str(node_vector_hash[id])) is c^(l-1)_v
            hash_l_list = [hash(str(node_vector_hash[id]))]

            for i in range(k):  # iterate pos of neighbor
                base_power = pow(n, k - 1 - i)
                id_remain = id - node_to_id[node_vector[i]] * base_power

                # hash_neighbor_list is {{c^l_(v,i)}}
                hash_neighbor_list = []
                for node in node_list:  # iterate neighbor node
                    id_cur = id_remain + node_to_id[node] * base_power
                    hash_neighbor_list.append(node_vector_hash[id_cur])
                hash_neighbor_list.sort()

                hash_l_list.append(hash(str(hash_neighbor_list)))
            # print('hash_l_list', hash_l_list)
            hash_l = hash(str(hash_l_list))
            if hash_l not in hash_wl_discret:
                hash_wl_discret[hash_l] = cnt
                cnt += 1

            node_vector_hash_nxt.append(hash_l)
            node_vector_hash_discret_nxt.append(hash_wl_discret[hash_l])

        if debug:
            counter = Counter(node_vector_hash_discret)
            print(sorted(counter.items(), key=lambda x: x[1]))

        if hash(str(node_vector_hash_discret)) == hash(
            str(node_vector_hash_discret_nxt)
        ):
            counter = Counter(node_vector_hash_nxt)
            return_hash = sorted(counter.items(), key=lambda x: x[0])
            if debug:
                print("return", hash(str(return_hash)))
            return hash(str(return_hash))

        node_vector_hash = copy.deepcopy(node_vector_hash_nxt)
        node_vector_hash_discret = copy.deepcopy(node_vector_hash_discret_nxt)


def wl_hash_tqdm(G, k, debug=False):
    # k should satisfy that any subgraph with k nodes is distinguishable with 1-WL
    node_list = [x for x in G.nodes]
    # print('node_list', node_list)
    n = G.number_of_nodes()
    node_to_id = dict()
    cnt = 0
    for i in range(n):
        node_to_id[node_list[i]] = i
    hash_wl_discret = dict()  # discretization of wl hash value
    node_vector_hash = []  # hash_value for each vector
    node_vector_hash_discret = []  # hash_value discretization for each vector
    node_vector_list = list(product(node_list, repeat=k))

    if debug:
        print("preprocessing---")
        for node_vector in tqdm(node_vector_list):
            sub_G = G.subgraph(node_vector)
            hash_wl = nx.weisfeiler_lehman_graph_hash(sub_G)
            # hash_wl = sub_G.number_of_edges()
            if hash_wl not in hash_wl_discret:
                hash_wl_discret[hash_wl] = cnt
                cnt += 1
            node_vector_hash.append(hash_wl)
            node_vector_hash_discret.append(hash_wl_discret[hash_wl])
    else:
        for node_vector in node_vector_list:
            sub_G = G.subgraph(node_vector)
            hash_wl = nx.weisfeiler_lehman_graph_hash(sub_G)
            # hash_wl = sub_G.number_of_edges()
            if hash_wl not in hash_wl_discret:
                hash_wl_discret[hash_wl] = cnt
                cnt += 1
            node_vector_hash.append(hash_wl)
            node_vector_hash_discret.append(hash_wl_discret[hash_wl])

    epoch = 1
    while epoch:
        if debug:
            print(f"Epoch {epoch}:")
        epoch += 1
        hash_wl_discret = dict()
        node_vector_hash_nxt = []
        node_vector_hash_discret_nxt = []
        cnt = 0
        if debug:
            for id in tqdm(range(len(node_vector_list))):
                node_vector = node_vector_list[id]

                # hash_l_list is c^l_v, hash(str(node_vector_hash[id])) is c^(l-1)_v
                hash_l_list = [hash(str(node_vector_hash[id]))]

                for i in range(k):  # iterate pos of neighbor
                    base_power = pow(n, k - 1 - i)
                    id_remain = id - node_to_id[node_vector[i]] * base_power

                    # hash_neighbor_list is {{c^l_(v,i)}}
                    hash_neighbor_list = []
                    for node in node_list:  # iterate neighbor node
                        id_cur = id_remain + node_to_id[node] * base_power
                        hash_neighbor_list.append(node_vector_hash[id_cur])
                    hash_neighbor_list.sort()

                    hash_l_list.append(hash(str(hash_neighbor_list)))
                # print('hash_l_list', hash_l_list)
                hash_l = hash(str(hash_l_list))
                if hash_l not in hash_wl_discret:
                    hash_wl_discret[hash_l] = cnt
                    cnt += 1

                node_vector_hash_nxt.append(hash_l)
                node_vector_hash_discret_nxt.append(hash_wl_discret[hash_l])
        else:
            for id in range(len(node_vector_list)):
                node_vector = node_vector_list[id]

                # hash_l_list is c^l_v, hash(str(node_vector_hash[id])) is c^(l-1)_v
                hash_l_list = [hash(str(node_vector_hash[id]))]

                for i in range(k):  # iterate pos of neighbor
                    base_power = pow(n, k - 1 - i)
                    id_remain = id - node_to_id[node_vector[i]] * base_power

                    # hash_neighbor_list is {{c^l_(v,i)}}
                    hash_neighbor_list = []
                    for node in node_list:  # iterate neighbor node
                        id_cur = id_remain + node_to_id[node] * base_power
                        hash_neighbor_list.append(node_vector_hash[id_cur])
                    hash_neighbor_list.sort()

                    hash_l_list.append(hash(str(hash_neighbor_list)))
                # print('hash_l_list', hash_l_list)
                hash_l = hash(str(hash_l_list))
                if hash_l not in hash_wl_discret:
                    hash_wl_discret[hash_l] = cnt
                    cnt += 1

                node_vector_hash_nxt.append(hash_l)
                node_vector_hash_discret_nxt.append(hash_wl_discret[hash_l])

        if debug:
            counter = Counter(node_vector_hash_discret)
            print(sorted(counter.items(), key=lambda x: x[1]))

        if hash(str(node_vector_hash_discret)) == hash(
            str(node_vector_hash_discret_nxt)
        ):
            counter = Counter(node_vector_hash_nxt)
            return_hash = sorted(counter.items(), key=lambda x: x[0])
            if debug:
                print("return", hash(str(return_hash)))
            return hash(str(return_hash))

        node_vector_hash = copy.deepcopy(node_vector_hash_nxt)
        node_vector_hash_discret = copy.deepcopy(node_vector_hash_discret_nxt)


def fwl_hash_tqdm(G, k, debug=False):
    # k should satisfy that any subgraph with k nodes is distinguishable with 1-WL
    node_list = [x for x in G.nodes]
    # print('node_list', node_list)
    n = G.number_of_nodes()
    node_to_id = dict()
    cnt = 0
    for i in range(n):
        node_to_id[node_list[i]] = i
    hash_wl_discret = dict()  # discretization of wl hash value
    node_vector_hash = []  # hash_value for each vector
    node_vector_hash_discret = []  # hash_value discretization for each vector
    node_vector_list = list(product(node_list, repeat=k))

    if debug:
        print("preprocessing---")
        for node_vector in tqdm(node_vector_list):
            sub_G = G.subgraph(node_vector)
            hash_wl = nx.weisfeiler_lehman_graph_hash(sub_G)
            # hash_wl = sub_G.number_of_edges()
            if hash_wl not in hash_wl_discret:
                hash_wl_discret[hash_wl] = cnt
                cnt += 1
            node_vector_hash.append(hash_wl)
            node_vector_hash_discret.append(hash_wl_discret[hash_wl])
    else:
        for node_vector in node_vector_list:
            sub_G = G.subgraph(node_vector)
            hash_wl = nx.weisfeiler_lehman_graph_hash(sub_G)
            # hash_wl = sub_G.number_of_edges()
            if hash_wl not in hash_wl_discret:
                hash_wl_discret[hash_wl] = cnt
                cnt += 1
            node_vector_hash.append(hash_wl)
            node_vector_hash_discret.append(hash_wl_discret[hash_wl])

    epoch = 1
    while epoch:
        if debug:
            print(f"Epoch {epoch}:")
        epoch += 1
        hash_wl_discret = dict()
        node_vector_hash_nxt = []
        node_vector_hash_discret_nxt = []
        cnt = 0
        if debug:
            for id in tqdm(range(len(node_vector_list))):
                node_vector = node_vector_list[id]

                # hash_l_list is c^l_v, hash(str(node_vector_hash[id])) is c^(l-1)_v
                # hash_l_list = [hash(str(node_vector_hash[id]))]
                hash_l_list = []

                for node in node_list:  # iterate neighbor node
                    id_node = node_to_id[node]
                    hash_neighbor_list = []

                    for i in range(k):  # iterate pos of neighbor
                        base_power = pow(n, k - 1 - i)
                        id_cur = (
                            id + (id_node - node_to_id[node_vector[i]]) * base_power
                        )
                        hash_neighbor_list.append(node_vector_hash[id_cur])

                    hash_l_list.append(hash(str(hash_neighbor_list)))

                counter = Counter(hash_l_list)
                hash_l_list_sorted = sorted(counter.items(), key=lambda x: x[0])
                hash_l_list_sorted.append(node_vector_hash[id])
                hash_l = hash(str(hash_l_list_sorted))
                if hash_l not in hash_wl_discret:
                    hash_wl_discret[hash_l] = cnt
                    cnt += 1

                node_vector_hash_nxt.append(hash_l)
                node_vector_hash_discret_nxt.append(hash_wl_discret[hash_l])
        else:
            for id in range(len(node_vector_list)):
                node_vector = node_vector_list[id]

                # hash_l_list is c^l_v, hash(str(node_vector_hash[id])) is c^(l-1)_v
                # hash_l_list = [hash(str(node_vector_hash[id]))]
                hash_l_list = []

                for node in node_list:  # iterate neighbor node
                    id_node = node_to_id[node]
                    hash_neighbor_list = []

                    for i in range(k):  # iterate pos of neighbor
                        base_power = pow(n, k - 1 - i)
                        id_cur = (
                            id + (id_node - node_to_id[node_vector[i]]) * base_power
                        )
                        hash_neighbor_list.append(node_vector_hash[id_cur])

                    hash_l_list.append(hash(str(hash_neighbor_list)))

                counter = Counter(hash_l_list)
                hash_l_list_sorted = sorted(counter.items(), key=lambda x: x[0])
                hash_l_list_sorted.append(node_vector_hash[id])
                hash_l = hash(str(hash_l_list_sorted))
                if hash_l not in hash_wl_discret:
                    hash_wl_discret[hash_l] = cnt
                    cnt += 1

                node_vector_hash_nxt.append(hash_l)
                node_vector_hash_discret_nxt.append(hash_wl_discret[hash_l])

        # if debug:
        #     counter = Counter(node_vector_hash_discret)
        #     print(sorted(counter.items(), key=lambda x: x[1]))

        if hash(str(node_vector_hash_discret)) == hash(
            str(node_vector_hash_discret_nxt)
        ):
            counter = Counter(node_vector_hash_nxt)
            return_hash = sorted(counter.items(), key=lambda x: x[0])
            if debug:
                print("return", hash(str(return_hash)))
            return hash(str(return_hash))

        node_vector_hash = copy.deepcopy(node_vector_hash_nxt)
        node_vector_hash_discret = copy.deepcopy(node_vector_hash_discret_nxt)


def wl_1_hash_tqdm(G, mode="none", debug=False):
    np.random.seed(2022)
    random.seed(2022)
    # k should satisfy that any subgraph with k nodes is distinguishable with 1-WL
    node_list = [x for x in G.nodes]
    # print('node_list', node_list)
    n = G.number_of_nodes()
    node_to_id = dict()
    cnt = 0
    for i in range(n):
        node_to_id[node_list[i]] = i
    hash_wl_discret = dict()  # discretization of wl hash value
    node_hash = []  # hash_value for each node
    node_hash_discret = []  # hash_value discretization for each node

    if mode == "n1":
        for node in node_list:
            sub_nodes = list(nx.all_neighbors(G, node))
            # sub_nodes.append(node) # checking necessity!
            sub_G = G.subgraph(sub_nodes)
            # hash_wl = nx.weisfeiler_lehman_graph_hash(sub_G)
            hash_wl = str(nx.to_graph6_bytes(sub_G))
            if hash_wl not in hash_wl_discret:
                hash_wl_discret[hash_wl] = cnt
                cnt += 1
            node_hash.append(hash_wl)
            node_hash_discret.append(hash_wl_discret[hash_wl])
    elif mode == "n2":
        for node in node_list:
            sub_nodes = set(G.adj[node])
            sub_nodes_all = set(G.adj[node])
            sub_nodes_all.add(node)
            for sub_node in sub_nodes:
                sub_nodes_all.update(G.adj[sub_node])
            sub_G = G.subgraph(sub_nodes_all)
            # hash_wl = fwl_hash_tqdm(sub_G, 2)
            hash_wl = str(nx.to_graph6_bytes(sub_G))
            if hash_wl not in hash_wl_discret:
                hash_wl_discret[hash_wl] = cnt
                cnt += 1
            node_hash.append(hash_wl)
            node_hash_discret.append(hash_wl_discret[hash_wl])
    elif mode == "s3":
        type_list = count_sub_3(G)
        for (id, node) in enumerate(node_list):
            hash_wl = hash(f"{type_list[0][id]}_{type_list[1][id]}")
            if hash_wl not in hash_wl_discret:
                hash_wl_discret[hash_wl] = cnt
                cnt += 1
            node_hash.append(hash_wl)
            node_hash_discret.append(hash_wl_discret[hash_wl])
    elif mode == "s4":
        type_list = count_sub_4(G)
        for (id, node) in enumerate(node_list):
            tmp_str = (
                f"{type_list[0][id]}_{type_list[1][id]}_{type_list[2][id]}_"
                + f"{type_list[3][id]}_{type_list[4][id]}_{type_list[5][id]}"
            )
            hash_wl = hash(tmp_str)
            if hash_wl not in hash_wl_discret:
                hash_wl_discret[hash_wl] = cnt
                cnt += 1
            node_hash.append(hash_wl)
            node_hash_discret.append(hash_wl_discret[hash_wl])
    elif mode == "s4_all":
        type_list = count_sub_4_all(G)
        for (id, node) in enumerate(node_list):
            tmp_str = (
                f"{type_list[0][id]}_{type_list[1][id]}_{type_list[2][id]}_"
                + f"{type_list[3][id]}_{type_list[4][id]}_{type_list[5][id]}"
            )
            hash_wl = hash(tmp_str)
            if hash_wl not in hash_wl_discret:
                hash_wl_discret[hash_wl] = cnt
                cnt += 1
            node_hash.append(hash_wl)
            node_hash_discret.append(hash_wl_discret[hash_wl])
    elif mode == "m1":
        hash_return = []
        for i in range(n):
            hash_return.append(wl_m_tqdm(G, [i], debug=debug))
        counter = Counter(hash_return)
        hash_return_sorted = sorted(counter.items(), key=lambda x: x[0])
        hash_return_sorted.append(wl_1_hash_tqdm(G))
        return_hash = hash(str(hash_return_sorted))
        return return_hash
    elif mode == "m2":
        hash_return = []
        for i in range(n):
            for j in range(i + 1, n):
                hash_return.append(wl_m_tqdm(G, [i, j], debug=debug))
        counter = Counter(hash_return)
        hash_return_sorted = sorted(counter.items(), key=lambda x: x[0])
        hash_return_sorted.append(wl_1_hash_tqdm(G))
        return_hash = hash(str(hash_return_sorted))
        return return_hash
    elif mode == "none" or mode == "None":
        for node in node_list:
            # hash_wl = 1
            hash_wl = G.degree[node]
            if hash_wl not in hash_wl_discret:
                hash_wl_discret[hash_wl] = cnt
                cnt += 1
            node_hash.append(hash_wl)
            node_hash_discret.append(hash_wl_discret[hash_wl])
    else:
        print(f"{mode} is not supported!")
        exit()

    epoch = 1
    while epoch:
        if debug:
            print(f"Epoch {epoch}:")
        epoch += 1
        hash_wl_discret = dict()
        node_hash_nxt = []
        node_hash_discret_nxt = []
        cnt = 0
        if debug:
            for id in tqdm(range(len(node_list))):
                node = node_list[id]
                hash_neighbor_list = []

                for neighbor in nx.all_neighbors(G, node):
                    hash_neighbor_list.append(node_hash[node_to_id[neighbor]])

                counter = Counter(hash_neighbor_list)
                hash_list = sorted(counter.items(), key=lambda x: x[0])
                hash_list.append(node_hash[id])
                hash_l = hash(str(hash_list))
                if hash_l not in hash_wl_discret:
                    hash_wl_discret[hash_l] = cnt
                    cnt += 1
                node_hash_nxt.append(hash_l)
                node_hash_discret_nxt.append(hash_wl_discret[hash_l])
        else:
            for id in range(len(node_list)):
                node = node_list[id]
                hash_neighbor_list = []

                for neighbor in nx.all_neighbors(G, node):
                    hash_neighbor_list.append(node_hash[node_to_id[neighbor]])

                counter = Counter(hash_neighbor_list)
                hash_list = sorted(counter.items(), key=lambda x: x[0])
                hash_list.append(node_hash[id])
                hash_l = hash(str(hash_list))
                if hash_l not in hash_wl_discret:
                    hash_wl_discret[hash_l] = cnt
                    cnt += 1
                node_hash_nxt.append(hash_l)
                node_hash_discret_nxt.append(hash_wl_discret[hash_l])

        # if debug:
        #     counter = Counter(node_hash_discret)
        #     print(sorted(counter.items(), key=lambda x: x[1]))

        if hash(str(node_hash_discret)) == hash(str(node_hash_discret_nxt)):
            counter = Counter(node_hash_nxt)
            return_hash = sorted(counter.items(), key=lambda x: x[0])
            if debug:
                print("return", hash(str(return_hash)))
                print(sorted(counter.items(), key=lambda x: x[0]))
            return hash(str(return_hash))

        node_hash = copy.deepcopy(node_hash_nxt)
        node_hash_discret = copy.deepcopy(node_hash_discret_nxt)


def wl_m_tqdm(G, m, debug=False):
    # k should satisfy that any subgraph with k nodes is distinguishable with 1-WL
    node_list = [x for x in G.nodes]
    # print('node_list', node_list)
    n = G.number_of_nodes()
    node_to_id = dict()
    cnt = 0
    for i in range(n):
        node_to_id[node_list[i]] = i
    hash_wl_discret = dict()  # discretization of wl hash value
    node_hash = []  # hash_value for each node
    node_hash_discret = []  # hash_value discretization for each node

    for node in node_list:
        hash_wl = 1
        for (id, m_node) in enumerate(m):
            if node_to_id[node] == m_node:
                hash_wl = id + 2
        # hash_wl = G.degree[node]
        if hash_wl not in hash_wl_discret:
            hash_wl_discret[hash_wl] = cnt
            cnt += 1
        node_hash.append(hash_wl)
        node_hash_discret.append(hash_wl_discret[hash_wl])

    epoch = 1
    while epoch:
        if debug:
            print(f"Epoch {epoch}:")
        epoch += 1
        hash_wl_discret = dict()
        node_hash_nxt = []
        node_hash_discret_nxt = []
        cnt = 0
        if debug:
            for id in tqdm(range(len(node_list))):
                node = node_list[id]
                hash_neighbor_list = []

                for neighbor in nx.all_neighbors(G, node):
                    hash_neighbor_list.append(node_hash[node_to_id[neighbor]])

                counter = Counter(hash_neighbor_list)
                hash_list = sorted(counter.items(), key=lambda x: x[0])
                hash_list.append(node_hash[id])
                hash_l = hash(str(hash_list))
                if hash_l not in hash_wl_discret:
                    hash_wl_discret[hash_l] = cnt
                    cnt += 1
                node_hash_nxt.append(hash_l)
                node_hash_discret_nxt.append(hash_wl_discret[hash_l])
        else:
            for id in range(len(node_list)):
                node = node_list[id]
                hash_neighbor_list = []

                for neighbor in nx.all_neighbors(G, node):
                    hash_neighbor_list.append(node_hash[node_to_id[neighbor]])

                counter = Counter(hash_neighbor_list)
                hash_list = sorted(counter.items(), key=lambda x: x[0])
                hash_list.append(node_hash[id])
                hash_l = hash(str(hash_list))
                if hash_l not in hash_wl_discret:
                    hash_wl_discret[hash_l] = cnt
                    cnt += 1
                node_hash_nxt.append(hash_l)
                node_hash_discret_nxt.append(hash_wl_discret[hash_l])

        # if debug:
        #     counter = Counter(node_hash_discret)
        #     print(sorted(counter.items(), key=lambda x: x[1]))

        if hash(str(node_hash_discret)) == hash(str(node_hash_discret_nxt)):
            counter = Counter(node_hash_nxt)
            return_hash = sorted(counter.items(), key=lambda x: x[0])
            if debug:
                print("return", hash(str(return_hash)))
                print(sorted(counter.items(), key=lambda x: x[0]))
            return hash(str(return_hash))

        node_hash = copy.deepcopy(node_hash_nxt)
        node_hash_discret = copy.deepcopy(node_hash_discret_nxt)


def count_sub_3(G):
    node_list = list(G.nodes)
    n = len(node_list)
    node_to_id = dict()
    for i in range(n):
        node_to_id[node_list[i]] = i
    # 0 type is triangle, 1 type is 3-path
    type_list = [[0] * n, [0] * n]
    for (id_1, node_1) in enumerate(node_list):
        for node_2 in G.adj[node_1]:
            id_2 = node_to_id[node_2]
            for node_3 in G.adj[node_2]:
                id_3 = node_to_id[node_3]
                if id_3 == id_1:
                    continue
                if G.has_edge(node_1, node_3):  # triangle
                    type_list[0][id_1] += 1
                    type_list[0][id_2] += 1
                    type_list[0][id_3] += 1
                else:  # 3-path
                    type_list[1][id_1] += 1
                    type_list[1][id_2] += 1
                    type_list[1][id_3] += 1
    return type_list


def count_sub_4(G):
    """
    type 0:     o - o - o - o

    type 1:     o - o - o
                    |
                    o

    type 2:     o - o
                | / |
                o - o

    type 3:     o - o
                | x |
                o - o

    type 4:     o - o
                |   |
                o - o

    type 5:     o - o - o
                    | /
                    o


    """
    node_list = list(G.nodes)
    n = len(node_list)
    node_to_id = dict()
    for i in range(n):
        node_to_id[node_list[i]] = i
    type_list = [[0] * n, [0] * n, [0] * n, [0] * n, [0] * n, [0] * n]
    for (id_1, node_1) in enumerate(node_list):
        for node_2 in G.adj[node_1]:
            id_2 = node_to_id[node_2]
            for node_3 in G.adj[node_2]:
                id_3 = node_to_id[node_3]
                if id_3 == id_1:
                    continue
                for node_4 in G.adj[node_3]:
                    id_4 = node_to_id[node_4]
                    if id_4 == id_1 or id_4 == id_2:
                        continue
                    edge_num = (
                        (node_1 in G[node_3])
                        + (node_1 in G[node_4])
                        + (node_2 in G[node_4])
                    )
                    if edge_num != 1:
                        type_list[edge_num][id_1] += 1
                        type_list[edge_num][id_2] += 1
                        type_list[edge_num][id_3] += 1
                        type_list[edge_num][id_4] += 1
                    elif G.has_edge(node_1, node_4):
                        type_list[4][id_1] += 1
                        type_list[4][id_2] += 1
                        type_list[4][id_3] += 1
                        type_list[4][id_4] += 1
                    else:
                        type_list[5][id_1] += 1
                        type_list[5][id_2] += 1
                        type_list[5][id_3] += 1
                        type_list[5][id_4] += 1
    for (id_1, node_1) in enumerate(node_list):
        for node_2 in G.adj[node_1]:
            id_2 = node_to_id[node_2]
            for node_3 in G.adj[node_1]:
                id_3 = node_to_id[node_3]
                if id_3 <= id_2:
                    continue
                for node_4 in G.adj[node_1]:
                    id_4 = node_to_id[node_4]
                    if id_4 <= id_3:
                        continue
                    edge_num = (
                        (node_2 in G[node_3])
                        + (node_3 in G[node_4])
                        + (node_2 in G[node_4])
                    )
                    if edge_num == 0:
                        type_list[1][id_1] += 1
                        type_list[1][id_2] += 1
                        type_list[1][id_3] += 1
                        type_list[1][id_4] += 1
    return type_list


def count_sub_4_all(G):
    """
    type 0:     o - o - o - o

    type 1:     o - o - o
                    |
                    o

    type 2:     o - o
                | / |
                o - o

    type 3:     o - o
                | x |
                o - o

    type 4:     o - o
                |   |
                o - o

    type 5:     o - o - o
                    | /
                    o


    """
    node_list = list(G.nodes)
    n = len(node_list)
    node_to_id = dict()
    for i in range(n):
        node_to_id[node_list[i]] = i
    type_list = [[0] * n, [0] * n, [0] * n, [0] * n, [0] * n, [0] * n]
    id = [0, 0, 0, 0]
    node = [0, 0, 0, 0]
    for (id[0], node[0]) in enumerate(node_list):
        for (id[1], node[1]) in enumerate(node_list):
            if id[0] == id[1]:
                continue
            for (id[2], node[2]) in enumerate(node_list):
                if id[0] == id[2] or id[1] == id[2]:
                    continue
                for (id[3], node[3]) in enumerate(node_list):
                    if id[0] == id[3] or id[1] == id[3] or id[1] == id[3]:
                        continue
                    edge_num = (
                        (node[0] in G[node[1]])
                        + (node[0] in G[node[2]])
                        + (node[0] in G[node[3]])
                        + (node[1] in G[node[2]])
                        + (node[1] in G[node[3]])
                        + (node[2] in G[node[3]])
                    )
                    if edge_num < 3:
                        continue
                    elif edge_num > 4:
                        for i in range(4):
                            type_list[edge_num - 3][id[i]] += 1
                    else:
                        real_type_num = (edge_num - 3) * 4
                        check = False
                        for i in range(4):
                            neighbor_num = 0
                            for j in range(4):
                                neighbor_num += node[i] in G[node[j]]
                            if neighbor_num == 3:
                                check = True
                                break
                        if check:
                            for i in range(4):
                                type_list[real_type_num + 1][id[i]] += 1
                        else:
                            for i in range(4):
                                type_list[real_type_num][id[i]] += 1

    return type_list


def graph6_to_pyg(x):
    return from_networkx(nx.from_graph6_bytes(x.encode()))


def pyg_to_graph6(x):
    return to_networkx(x, to_undirected=True)


def rooks_graph(m, n):
    G = nx.complete_graph(m)
    H = nx.complete_graph(n)
    Rooks = nx.cartesian_product(G, H)
    return Rooks


def Hamming_weight(n):
    n = (n & 0x55555555) + ((n >> 1) & 0x55555555)
    n = (n & 0x33333333) + ((n >> 2) & 0x33333333)
    n = (n & 0x0F0F0F0F) + ((n >> 4) & 0x0F0F0F0F)
    n = (n & 0x00FF00FF) + ((n >> 8) & 0x00FF00FF)
    n = (n & 0x0000FFFF) + ((n >> 16) & 0x0000FFFF)
    return n


def construct_code_graph(node_list):
    G = nx.Graph()
    G.add_nodes_from(node_list)
    for i in range(len(node_list)):
        node_i = int(node_list[i], 2)
        for j in range(i, len(node_list)):
            node_j = int(node_list[j], 2)
            if Hamming_weight(node_i ^ node_j) == 2:
                G.add_edge(node_list[i], node_list[j])
    return G


def shrikhande_graph():
    node_list = [
        "000000",
        "110000",
        "110110",
        "000110",
        "001100",
        "011000",
        "111010",
        "101110",
        "101101",
        "011101",
        "011011",
        "101011",
        "100001",
        "110101",
        "010111",
        "000011",
    ]
    return construct_code_graph(node_list)


def csl_graph(m, r):
    if gcd(m, r) != 1:
        print(f"csl graph should satisfy m({m}) and r({r}) co-prime!")
        return None
    G = nx.cycle_graph(m)
    for start_pos in range(r):
        edge_list = [(x, (x + r) % m) for x in range(start_pos, m, r)]
        G.add_edges_from(edge_list)
    return G


def WL_hash(G, k, mode=None, debug=False):
    # k should satisfy that any subgraph with k nodes is distinguishable with 1-WL
    node_list = [x for x in G.nodes]
    # print('node_list', node_list)
    n = G.number_of_nodes()
    node_to_id = dict()
    cnt = 0
    for i in range(n):
        node_to_id[node_list[i]] = i
    hash_wl_discret = dict()  # discretization of wl hash value
    node_vector_hash = []  # hash_value for each vector
    node_vector_hash_discret = []  # hash_value discretization for each vector
    node_vector_list = list(product(node_list, repeat=k))

    if debug:
        print("preprocessing---")
        for node_vector in tqdm(node_vector_list):
            sub_G = G.subgraph(node_vector)
            hash_wl = nx.weisfeiler_lehman_graph_hash(sub_G)
            # hash_wl = sub_G.number_of_edges()
            if hash_wl not in hash_wl_discret:
                hash_wl_discret[hash_wl] = cnt
                cnt += 1
            node_vector_hash.append(hash_wl)
            node_vector_hash_discret.append(hash_wl_discret[hash_wl])
    else:
        for node_vector in node_vector_list:
            sub_G = G.subgraph(node_vector)
            hash_wl = nx.weisfeiler_lehman_graph_hash(sub_G)
            # hash_wl = sub_G.number_of_edges()
            if hash_wl not in hash_wl_discret:
                hash_wl_discret[hash_wl] = cnt
                cnt += 1
            node_vector_hash.append(hash_wl)
            node_vector_hash_discret.append(hash_wl_discret[hash_wl])

    epoch = 1
    while epoch:
        if debug:
            print(f"Epoch {epoch}:")
        epoch += 1
        hash_wl_discret = dict()
        node_vector_hash_nxt = []
        node_vector_hash_discret_nxt = []
        cnt = 0
        if debug:
            for id in tqdm(range(len(node_vector_list))):
                node_vector = node_vector_list[id]

                # hash_l_list is c^l_v, hash(str(node_vector_hash[id])) is c^(l-1)_v
                hash_l_list = [hash(str(node_vector_hash[id]))]

                for i in range(k):  # iterate pos of neighbor
                    base_power = pow(n, k - 1 - i)
                    id_remain = id - node_to_id[node_vector[i]] * base_power

                    # hash_neighbor_list is {{c^l_(v,i)}}
                    hash_neighbor_list = []
                    for node in node_list:  # iterate neighbor node
                        id_cur = id_remain + node_to_id[node] * base_power
                        hash_neighbor_list.append(node_vector_hash[id_cur])
                    hash_neighbor_list.sort()

                    hash_l_list.append(hash(str(hash_neighbor_list)))
                # print('hash_l_list', hash_l_list)
                hash_l = hash(str(hash_l_list))
                if hash_l not in hash_wl_discret:
                    hash_wl_discret[hash_l] = cnt
                    cnt += 1

                node_vector_hash_nxt.append(hash_l)
                node_vector_hash_discret_nxt.append(hash_wl_discret[hash_l])
        else:
            for id in range(len(node_vector_list)):
                node_vector = node_vector_list[id]

                # hash_l_list is c^l_v, hash(str(node_vector_hash[id])) is c^(l-1)_v
                hash_l_list = [hash(str(node_vector_hash[id]))]

                for i in range(k):  # iterate pos of neighbor
                    base_power = pow(n, k - 1 - i)
                    id_remain = id - node_to_id[node_vector[i]] * base_power

                    # hash_neighbor_list is {{c^l_(v,i)}}
                    hash_neighbor_list = []
                    for node in node_list:  # iterate neighbor node
                        id_cur = id_remain + node_to_id[node] * base_power
                        hash_neighbor_list.append(node_vector_hash[id_cur])
                    hash_neighbor_list.sort()

                    hash_l_list.append(hash(str(hash_neighbor_list)))
                # print('hash_l_list', hash_l_list)
                hash_l = hash(str(hash_l_list))
                if hash_l not in hash_wl_discret:
                    hash_wl_discret[hash_l] = cnt
                    cnt += 1

                node_vector_hash_nxt.append(hash_l)
                node_vector_hash_discret_nxt.append(hash_wl_discret[hash_l])

        if debug:
            counter = Counter(node_vector_hash_discret)
            print(sorted(counter.items(), key=lambda x: x[1]))

        if hash(str(node_vector_hash_discret)) == hash(
            str(node_vector_hash_discret_nxt)
        ):
            counter = Counter(node_vector_hash_nxt)
            return_hash = sorted(counter.items(), key=lambda x: x[0])
            if debug:
                print("return", hash(str(return_hash)))
            return hash(str(return_hash))

        node_vector_hash = copy.deepcopy(node_vector_hash_nxt)
        node_vector_hash_discret = copy.deepcopy(node_vector_hash_discret_nxt)


def FWL_hash(G, k, mode=None, debug=False):
    # k should satisfy that any subgraph with k nodes is distinguishable with 1-WL
    node_list = [x for x in G.nodes]
    # print('node_list', node_list)
    n = G.number_of_nodes()
    node_to_id = dict()
    cnt = 0
    for i in range(n):
        node_to_id[node_list[i]] = i
    hash_wl_discret = dict()  # discretization of wl hash value
    node_vector_hash = []  # hash_value for each vector
    node_vector_hash_discret = []  # hash_value discretization for each vector
    node_vector_list = list(product(node_list, repeat=k))

    if debug:
        print("preprocessing---")
        for node_vector in tqdm(node_vector_list):
            sub_G = G.subgraph(node_vector)
            hash_wl = nx.weisfeiler_lehman_graph_hash(sub_G)
            # hash_wl = sub_G.number_of_edges()
            if hash_wl not in hash_wl_discret:
                hash_wl_discret[hash_wl] = cnt
                cnt += 1
            node_vector_hash.append(hash_wl)
            node_vector_hash_discret.append(hash_wl_discret[hash_wl])
    else:
        for node_vector in node_vector_list:
            sub_G = G.subgraph(node_vector)
            hash_wl = nx.weisfeiler_lehman_graph_hash(sub_G)
            # hash_wl = sub_G.number_of_edges()
            if hash_wl not in hash_wl_discret:
                hash_wl_discret[hash_wl] = cnt
                cnt += 1
            node_vector_hash.append(hash_wl)
            node_vector_hash_discret.append(hash_wl_discret[hash_wl])

    epoch = 1
    while epoch:
        if debug:
            print(f"Epoch {epoch}:")
        epoch += 1
        hash_wl_discret = dict()
        node_vector_hash_nxt = []
        node_vector_hash_discret_nxt = []
        cnt = 0
        if debug:
            for id in tqdm(range(len(node_vector_list))):
                node_vector = node_vector_list[id]

                # hash_l_list is c^l_v, hash(str(node_vector_hash[id])) is c^(l-1)_v
                # hash_l_list = [hash(str(node_vector_hash[id]))]
                hash_l_list = []

                for node in node_list:  # iterate neighbor node
                    id_node = node_to_id[node]
                    hash_neighbor_list = []

                    for i in range(k):  # iterate pos of neighbor
                        base_power = pow(n, k - 1 - i)
                        id_cur = (
                            id + (id_node - node_to_id[node_vector[i]]) * base_power
                        )
                        hash_neighbor_list.append(node_vector_hash[id_cur])

                    hash_l_list.append(hash(str(hash_neighbor_list)))

                counter = Counter(hash_l_list)
                hash_l_list_sorted = sorted(counter.items(), key=lambda x: x[0])
                hash_l_list_sorted.append(node_vector_hash[id])
                hash_l = hash(str(hash_l_list_sorted))
                if hash_l not in hash_wl_discret:
                    hash_wl_discret[hash_l] = cnt
                    cnt += 1

                node_vector_hash_nxt.append(hash_l)
                node_vector_hash_discret_nxt.append(hash_wl_discret[hash_l])
        else:
            for id in range(len(node_vector_list)):
                node_vector = node_vector_list[id]

                # hash_l_list is c^l_v, hash(str(node_vector_hash[id])) is c^(l-1)_v
                # hash_l_list = [hash(str(node_vector_hash[id]))]
                hash_l_list = []

                for node in node_list:  # iterate neighbor node
                    id_node = node_to_id[node]
                    hash_neighbor_list = []

                    for i in range(k):  # iterate pos of neighbor
                        base_power = pow(n, k - 1 - i)
                        id_cur = (
                            id + (id_node - node_to_id[node_vector[i]]) * base_power
                        )
                        hash_neighbor_list.append(node_vector_hash[id_cur])

                    hash_l_list.append(hash(str(hash_neighbor_list)))

                counter = Counter(hash_l_list)
                hash_l_list_sorted = sorted(counter.items(), key=lambda x: x[0])
                hash_l_list_sorted.append(node_vector_hash[id])
                hash_l = hash(str(hash_l_list_sorted))
                if hash_l not in hash_wl_discret:
                    hash_wl_discret[hash_l] = cnt
                    cnt += 1

                node_vector_hash_nxt.append(hash_l)
                node_vector_hash_discret_nxt.append(hash_wl_discret[hash_l])

        # if debug:
        #     counter = Counter(node_vector_hash_discret)
        #     print(sorted(counter.items(), key=lambda x: x[1]))

        if hash(str(node_vector_hash_discret)) == hash(
            str(node_vector_hash_discret_nxt)
        ):
            counter = Counter(node_vector_hash_nxt)
            return_hash = sorted(counter.items(), key=lambda x: x[0])
            if debug:
                print("return", hash(str(return_hash)))
            return hash(str(return_hash))

        node_vector_hash = copy.deepcopy(node_vector_hash_nxt)
        node_vector_hash_discret = copy.deepcopy(node_vector_hash_discret_nxt)


def WL_1_hash(G, k=1, mode="none", debug=False):
    np.random.seed(2022)
    random.seed(2022)
    # k should satisfy that any subgraph with k nodes is distinguishable with 1-WL
    node_list = [x for x in G.nodes]
    # print('node_list', node_list)
    n = G.number_of_nodes()
    node_to_id = dict()
    cnt = 0
    for i in range(n):
        node_to_id[node_list[i]] = i
    hash_wl_discret = dict()  # discretization of wl hash value
    node_hash = []  # hash_value for each node
    node_hash_discret = []  # hash_value discretization for each node
    # iso_type_list = []  # an example g6 for each iso type

    if mode.startswith("de"):
        radius = int(re.findall(r"\d+", mode)[0])
        # print(radius)
        for node in node_list:
            sub_G = nx.ego_graph(G, node, radius)
            hash_wl = subgraph_de(sub_G, node)
            if hash_wl not in hash_wl_discret:
                hash_wl_discret[hash_wl] = cnt
                cnt += 1
            node_hash.append(hash_wl)
            node_hash_discret.append(hash_wl_discret[hash_wl])
    elif mode == "s3":
        type_list = count_sub_3(G)
        for (id, node) in enumerate(node_list):
            hash_wl = hash(f"{type_list[0][id]}_{type_list[1][id]}")
            if hash_wl not in hash_wl_discret:
                hash_wl_discret[hash_wl] = cnt
                cnt += 1
            node_hash.append(hash_wl)
            node_hash_discret.append(hash_wl_discret[hash_wl])
    elif mode == "s4":
        type_list = count_sub_4(G)
        for (id, node) in enumerate(node_list):
            tmp_str = (
                f"{type_list[0][id]}_{type_list[1][id]}_{type_list[2][id]}_"
                + f"{type_list[3][id]}_{type_list[4][id]}_{type_list[5][id]}"
            )
            hash_wl = hash(tmp_str)
            if hash_wl not in hash_wl_discret:
                hash_wl_discret[hash_wl] = cnt
                cnt += 1
            node_hash.append(hash_wl)
            node_hash_discret.append(hash_wl_discret[hash_wl])
    elif mode == "s4_all":
        type_list = count_sub_4_all(G)
        for (id, node) in enumerate(node_list):
            tmp_str = (
                f"{type_list[0][id]}_{type_list[1][id]}_{type_list[2][id]}_"
                + f"{type_list[3][id]}_{type_list[4][id]}_{type_list[5][id]}"
            )
            hash_wl = hash(tmp_str)
            if hash_wl not in hash_wl_discret:
                hash_wl_discret[hash_wl] = cnt
                cnt += 1
            node_hash.append(hash_wl)
            node_hash_discret.append(hash_wl_discret[hash_wl])
    elif mode == "m1":
        hash_return = []
        for i in range(n):
            hash_return.append(wl_m_tqdm(G, [i], debug=debug))
        counter = Counter(hash_return)
        hash_return_sorted = sorted(counter.items(), key=lambda x: x[0])
        hash_return_sorted.append(wl_1_hash_tqdm(G))
        return_hash = hash(str(hash_return_sorted))
        return return_hash
    elif mode == "m2":
        hash_return = []
        for i in range(n):
            for j in range(i + 1, n):
                hash_return.append(wl_m_tqdm(G, [i, j], debug=debug))
        counter = Counter(hash_return)
        hash_return_sorted = sorted(counter.items(), key=lambda x: x[0])
        hash_return_sorted.append(wl_1_hash_tqdm(G))
        return_hash = hash(str(hash_return_sorted))
        return return_hash
    elif mode == "r24":
        for node in node_list:
            sub_G = nx.ego_graph(G, node, 2, center=False)
            neighbor_1_hop_list = list(nx.all_neighbors(G, node))

            node_degree = len(neighbor_1_hop_list)
            cnt_r24 = 0
            node_dict = dict()
            node_dict_to_list = [[] for i in range(node_degree)]

            for neighbor_1_hop in neighbor_1_hop_list:
                node_dict[neighbor_1_hop] = cnt_r24
                cnt_r24 += 1

            for neighbor_1_hop in neighbor_1_hop_list:
                neighbor_2_hop_list = list(nx.all_neighbors(sub_G, neighbor_1_hop))
                for neighbor_2_hop in neighbor_2_hop_list:
                    if neighbor_2_hop not in node_dict:
                        node_dict[neighbor_2_hop] = cnt_r24
                        cnt_r24 += 1
                        node_dict_to_list.append([neighbor_1_hop])
                    else:
                        neighbor_2_hop_id = node_dict[neighbor_2_hop]
                        node_dict_to_list[neighbor_2_hop_id].append(neighbor_1_hop)

            edge_num_list = []
            for count_edge_subgraph in node_dict_to_list:
                edge_num_list.append(
                    nx.subgraph(sub_G, count_edge_subgraph).number_of_edges()
                )

            connected_neighbor = edge_num_list[0:node_degree]
            disconnected_neighbor = edge_num_list[node_degree:]

            counter_1 = Counter(connected_neighbor)
            counter_2 = Counter(disconnected_neighbor)
            hash_list_1 = sorted(counter_1.items(), key=lambda x: x[0])
            hash_list_2 = sorted(counter_2.items(), key=lambda x: x[0])

            hash_wl = hash(str(hash_list_1) + str(hash_list_2))
            if hash_wl not in hash_wl_discret:
                hash_wl_discret[hash_wl] = cnt
                cnt += 1
            node_hash.append(hash_wl)
            node_hash_discret.append(hash_wl_discret[hash_wl])
    elif mode == "none" or mode == "None":
        for node in node_list:
            # hash_wl = 1
            hash_wl = G.degree[node]
            if hash_wl not in hash_wl_discret:
                hash_wl_discret[hash_wl] = cnt
                cnt += 1
            node_hash.append(hash_wl)
            node_hash_discret.append(hash_wl_discret[hash_wl])
    elif mode.startswith("n"):
        radius = int(re.findall(r"\d+", mode)[0])
        # print(radius)
        for node in node_list:
            sub_G = nx.ego_graph(G, node, radius)
            hash_wl = pynauty.certificate(from_nx_to_nauty(sub_G))
            if hash_wl not in hash_wl_discret:
                hash_wl_discret[hash_wl] = cnt
                cnt += 1
            node_hash.append(hash_wl)
            node_hash_discret.append(hash_wl_discret[hash_wl])
    else:
        print(f"{mode} is not supported!")
        exit()
    epoch = 1
    while epoch:
        if debug:
            print(f"Epoch {epoch}:")
        epoch += 1
        hash_wl_discret = dict()
        node_hash_nxt = []
        node_hash_discret_nxt = []
        cnt = 0
        if debug:
            for id in tqdm(range(len(node_list))):
                node = node_list[id]
                hash_neighbor_list = []

                for neighbor in nx.all_neighbors(G, node):
                    hash_neighbor_list.append(node_hash[node_to_id[neighbor]])

                counter = Counter(hash_neighbor_list)
                hash_list = sorted(counter.items(), key=lambda x: x[0])
                hash_list.append(node_hash[id])
                hash_l = hash(str(hash_list))
                if hash_l not in hash_wl_discret:
                    hash_wl_discret[hash_l] = cnt
                    cnt += 1
                node_hash_nxt.append(hash_l)
                node_hash_discret_nxt.append(hash_wl_discret[hash_l])
        else:
            for id in range(len(node_list)):
                node = node_list[id]
                hash_neighbor_list = []

                for neighbor in nx.all_neighbors(G, node):
                    hash_neighbor_list.append(node_hash[node_to_id[neighbor]])

                counter = Counter(hash_neighbor_list)
                hash_list = sorted(counter.items(), key=lambda x: x[0])
                hash_list.append(node_hash[id])
                hash_l = hash(str(hash_list))
                if hash_l not in hash_wl_discret:
                    hash_wl_discret[hash_l] = cnt
                    cnt += 1
                node_hash_nxt.append(hash_l)
                node_hash_discret_nxt.append(hash_wl_discret[hash_l])

        # if debug:
        #     counter = Counter(node_hash_discret)
        #     print(sorted(counter.items(), key=lambda x: x[1]))

        if hash(str(node_hash_discret)) == hash(str(node_hash_discret_nxt)):
            counter = Counter(node_hash_nxt)
            return_hash = sorted(counter.items(), key=lambda x: x[0])
            if debug:
                print("return", hash(str(return_hash)))
                print(sorted(counter.items(), key=lambda x: x[0]))
            return hash(str(return_hash))

        node_hash = copy.deepcopy(node_hash_nxt)
        node_hash_discret = copy.deepcopy(node_hash_discret_nxt)


def is_k_isoregular(G, k):
    if k == 1:
        return nx.is_regular(G)
    if k == 2:
        return nx.is_strongly_regular(G)
    if k == 3:
        node_list = list(G.nodes())
        # n = len(node_list)
        type_dict = dict()
        subgraph_list = product(node_list, repeat=k)
        for subgraph_nodes in subgraph_list:
            subgraph = nx.subgraph(G, subgraph_nodes)
            neighbor = [list(G[node]) for node in subgraph_nodes]
            subgraph_type = nx.weisfeiler_lehman_graph_hash(subgraph)
            common_neighbor = len(
                list(set(neighbor[0]).intersection(neighbor[1], neighbor[2]))
            )
            if subgraph_type in type_dict:
                if type_dict[subgraph_type] != common_neighbor:
                    return False
            else:
                type_dict[subgraph_type] = common_neighbor
        return True
    else:
        raise NotImplementedError("Only k<=3 is available")


def param_of_4vtx_condition(G):
    if not nx.is_strongly_regular(G):
        raise TypeError("Only strongly regular may satisfy 4-vertex condition")
    node_list = list(G.nodes())
    subgraph_list = product(node_list, repeat=2)
    type_dict = dict()
    for subgraph_nodes in subgraph_list:
        if subgraph_nodes[0] == subgraph_nodes[1]:
            continue
        subgraph = nx.subgraph(G, subgraph_nodes)
        subgraph_type = nx.number_of_edges(subgraph)
        neighbor = [list(G[node]) for node in subgraph_nodes]
        common_neighbor = list(set(neighbor[0]).intersection(neighbor[1]))
        common_neighbor_edges = nx.subgraph(G, common_neighbor).number_of_edges()
        if subgraph_type in type_dict:
            if type_dict[subgraph_type] != common_neighbor_edges:
                return False
        else:
            type_dict[subgraph_type] = common_neighbor_edges
    return type_dict


def FWL_hashlib(G, k, mode=None, debug=False):
    # k should satisfy that any subgraph with k nodes is distinguishable with 1-WL
    node_list = [x for x in G.nodes]
    # print('node_list', node_list)
    n = G.number_of_nodes()
    node_to_id = dict()
    cnt = 0
    for i in range(n):
        node_to_id[node_list[i]] = i
    hash_wl_discret = dict()  # discretization of wl hash value
    node_vector_hash = []  # hash_value for each vector
    node_vector_hash_discret = []  # hash_value discretization for each vector
    node_vector_list = list(product(node_list, repeat=k))

    if debug:
        print("preprocessing---")
        for node_vector in tqdm(node_vector_list):
            sub_G = G.subgraph(node_vector)
            hash_wl = nx.weisfeiler_lehman_graph_hash(sub_G)
            # hash_wl = sub_G.number_of_edges()
            if hash_wl not in hash_wl_discret:
                hash_wl_discret[hash_wl] = cnt
                cnt += 1
            node_vector_hash.append(hash_wl)
            node_vector_hash_discret.append(hash_wl_discret[hash_wl])
    else:
        for node_vector in node_vector_list:
            sub_G = G.subgraph(node_vector)
            hash_wl = nx.weisfeiler_lehman_graph_hash(sub_G)
            # hash_wl = sub_G.number_of_edges()
            if hash_wl not in hash_wl_discret:
                hash_wl_discret[hash_wl] = cnt
                cnt += 1
            node_vector_hash.append(hash_wl)
            node_vector_hash_discret.append(hash_wl_discret[hash_wl])

    epoch = 1
    while epoch:
        if debug:
            print(f"Epoch {epoch}:")
        epoch += 1
        hash_wl_discret = dict()
        node_vector_hash_nxt = []
        node_vector_hash_discret_nxt = []
        cnt = 0
        if debug:
            for id in tqdm(range(len(node_vector_list))):
                node_vector = node_vector_list[id]

                # hash_l_list is c^l_v, hash(str(node_vector_hash[id])) is c^(l-1)_v
                # hash_l_list = [hash(str(node_vector_hash[id]))]
                hash_l_list = []

                for node in node_list:  # iterate neighbor node
                    id_node = node_to_id[node]
                    hash_neighbor_list = []

                    for i in range(k):  # iterate pos of neighbor
                        base_power = pow(n, k - 1 - i)
                        id_cur = (
                            id + (id_node - node_to_id[node_vector[i]]) * base_power
                        )
                        hash_neighbor_list.append(node_vector_hash[id_cur])

                    md5 = hashlib.md5()
                    md5.update(str(hash_neighbor_list).encode("utf-8"))
                    hash_l_list.append(md5.hexdigest())

                    # hash_l_list.append(hash(str(hash_neighbor_list)))

                counter = Counter(hash_l_list)
                hash_l_list_sorted = sorted(counter.items(), key=lambda x: x[0])
                hash_l_list_sorted.append(node_vector_hash[id])

                # hash_l = hash(str(hash_l_list_sorted))
                md5 = hashlib.md5()
                md5.update(str(hash_l_list_sorted).encode("utf-8"))
                hash_l = md5.hexdigest()

                if hash_l not in hash_wl_discret:
                    hash_wl_discret[hash_l] = cnt
                    cnt += 1

                node_vector_hash_nxt.append(hash_l)
                node_vector_hash_discret_nxt.append(hash_wl_discret[hash_l])
        else:
            for id in range(len(node_vector_list)):
                node_vector = node_vector_list[id]

                # hash_l_list is c^l_v, hash(str(node_vector_hash[id])) is c^(l-1)_v
                # hash_l_list = [hash(str(node_vector_hash[id]))]
                hash_l_list = []

                for node in node_list:  # iterate neighbor node
                    id_node = node_to_id[node]
                    hash_neighbor_list = []

                    for i in range(k):  # iterate pos of neighbor
                        base_power = pow(n, k - 1 - i)
                        id_cur = (
                            id + (id_node - node_to_id[node_vector[i]]) * base_power
                        )
                        hash_neighbor_list.append(node_vector_hash[id_cur])

                    md5 = hashlib.md5()
                    md5.update(str(hash_neighbor_list).encode("utf-8"))
                    hash_l_list.append(md5.hexdigest())

                    # hash_l_list.append(hash(str(hash_neighbor_list)))

                counter = Counter(hash_l_list)
                hash_l_list_sorted = sorted(counter.items(), key=lambda x: x[0])
                hash_l_list_sorted.append(node_vector_hash[id])

                # hash_l = hash(str(hash_l_list_sorted))
                md5 = hashlib.md5()
                md5.update(str(hash_l_list_sorted).encode("utf-8"))
                hash_l = md5.hexdigest()

                if hash_l not in hash_wl_discret:
                    hash_wl_discret[hash_l] = cnt
                    cnt += 1

                node_vector_hash_nxt.append(hash_l)
                node_vector_hash_discret_nxt.append(hash_wl_discret[hash_l])

        # if debug:
        #     counter = Counter(node_vector_hash_discret)
        #     print(sorted(counter.items(), key=lambda x: x[1]))

        if hash(str(node_vector_hash_discret)) == hash(
            str(node_vector_hash_discret_nxt)
        ):
            counter = Counter(node_vector_hash_nxt)
            return_hash = sorted(counter.items(), key=lambda x: x[0])
            md5 = hashlib.md5()
            md5.update(str(return_hash).encode("utf-8"))
            if debug:
                # print("return", hash(str(return_hash)))
                print("return", md5.hexdigest())
            return md5.hexdigest()

        node_vector_hash = copy.deepcopy(node_vector_hash_nxt)
        node_vector_hash_discret = copy.deepcopy(node_vector_hash_discret_nxt)


def incidence_graph(file):
    with open(file, "r") as f:
        design = [x.strip() for x in f.readlines()]
    g0 = np.zeros(shape=(len(design), len(design)), dtype=int)
    g = np.zeros(shape=(len(design), len(design)), dtype=int)
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            g[i, j] = int(design[i][j])
    graph_complete_up = np.concatenate([g0, g], axis=1)
    graph_complete_down = np.concatenate([g.T, g0], axis=1)
    graph_complete = np.concatenate([graph_complete_up, graph_complete_down], axis=0)
    graph = nx.from_numpy_array(graph_complete)
    # print(graph)
    return graph


def direct_graph(file):
    with open(file, "r") as f:
        graph = [x.strip() for x in f.readlines()]
    g = np.zeros(shape=(len(graph), len(graph)), dtype=int)
    for i in range(len(graph)):
        for j in range(len(graph)):
            g[i, j] = int(graph[i][j])
    return_graph = nx.from_numpy_array(g)
    return return_graph


def cll(n):
    g = nx.Graph()
    g.add_edges_from([(i, (i + 1) % n) for i in range(n)])
    g.add_edges_from([(i + n, (i + 1) % n + n) for i in range(n)])
    g.add_edges_from([(n * 2, i) for i in range(n * 2)])
    return g


def c2l(n):
    g = nx.cycle_graph(n * 2)
    g.add_edges_from([(n * 2, i) for i in range(n * 2)])
    return g


def from_nx_to_nauty(G):
    n = G.number_of_nodes()
    node_list = list(G.nodes)
    node_to_id = dict()
    for i in range(n):
        node_to_id[node_list[i]] = i
    adjacency_dict = dict()
    for k, v in dict(G.adj).items():
        adjacency_dict[node_to_id[k]] = [node_to_id[v_node] for v_node in v]
    # print(adjacency_dict)
    g = pynauty.Graph(number_of_vertices=n, adjacency_dict=adjacency_dict)
    return g


def subgraph_de(G, center, debug=False):
    np.random.seed(2022)
    random.seed(2022)
    node_list = [x for x in G.nodes]
    # print('node_list', node_list)
    n = G.number_of_nodes()
    node_to_id = dict()
    cnt = 0
    for i in range(n):
        node_to_id[node_list[i]] = i
    hash_wl_discret = dict()  # discretization of wl hash value
    node_hash = []  # hash_value for each node
    node_hash_discret = []  # hash_value discretization for each node

    for node in node_list:
        hash_wl = nx.shortest_path_length(G, center, node)
        if hash_wl not in hash_wl_discret:
            hash_wl_discret[hash_wl] = cnt
            cnt += 1
        node_hash.append(hash_wl)
        node_hash_discret.append(hash_wl_discret[hash_wl])

    epoch = 1
    while epoch:
        if debug:
            print(f"Epoch {epoch}:")
        hash_wl_discret = dict()
        node_hash_nxt = []
        node_hash_discret_nxt = []
        cnt = 0
        for id in tqdm(range(len(node_list))) if debug else range(len(node_list)):
            node = node_list[id]
            hash_neighbor_list = []

            for neighbor in nx.all_neighbors(G, node):
                hash_neighbor_list.append(node_hash[node_to_id[neighbor]])

            counter = Counter(hash_neighbor_list)
            hash_list = sorted(counter.items(), key=lambda x: x[0])
            hash_list.append(node_hash[id])
            hash_l = hash(str(hash_list))
            if hash_l not in hash_wl_discret:
                hash_wl_discret[hash_l] = cnt
                cnt += 1
            node_hash_nxt.append(hash_l)
            node_hash_discret_nxt.append(hash_wl_discret[hash_l])

        # if debug:
        #     counter = Counter(node_hash_discret)
        #     print(sorted(counter.items(), key=lambda x: x[1]))
        #     counter = Counter(node_hash)
        #     print(sorted(counter.items(), key=lambda x: x[0]))

        #     if write_node_hash:
        #         with open(node_hash_path / f"{epoch}.txt", "w") as f:
        #             for id in range(len(node_list)):
        #                 f.write(f"{id}: {node_hash[id]}\n")

        # if draw:
        #     g_draw = G.copy()
        #     attrs = dict()
        #     for id in range(len(node_list)):
        #         node = node_list[id]
        #         attrs[node] = {"color": from_discretid_to_color(node_hash_discret[id])}
        #     nx.set_node_attributes(g_draw, attrs)
        #     Path.mkdir(draw_path, exist_ok=True)
        #     nx.write_graphml(g_draw, draw_path / f"{epoch}.graphml", named_key_ids=True)

        if hash(str(node_hash_discret)) == hash(str(node_hash_discret_nxt)):
            counter = Counter(node_hash_nxt)
            return_hash = sorted(counter.items(), key=lambda x: x[0])
            if debug:
                print()
                print("return", hash(str(return_hash)))
                print(return_hash)
            #     if write_node_hash:
            #         with open(node_hash_path / f"{epoch + 1}.txt", "w") as f:
            #             for id in range(len(node_list)):
            #                 f.write(f"{id}: {node_hash_nxt[id]}\n")
            return hash(str(return_hash))

        node_hash = copy.deepcopy(node_hash_nxt)
        node_hash_discret = copy.deepcopy(node_hash_discret_nxt)

        epoch += 1

