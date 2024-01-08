import networkx as nx
import numpy as np
import random
from utils import fwl_hash_tqdm, wl_1_hash_tqdm, wl_hash_tqdm
import argparse
from tqdm import tqdm

random.seed(2022)
np.random.seed(2022)


def add_bool_arg(parser, name, default=False):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--" + name, dest=name, action="store_true")
    group.add_argument("--no-" + name, dest=name, action="store_false")
    parser.set_defaults(**{name: default})


parser = argparse.ArgumentParser(description="Create graph-8/9 a/c dataset.")
parser.add_argument("--ac", type=str, default="a")
parser.add_argument("--node", type=int, default="8")
parser.add_argument("--wl", type=int, default="2")
parser.add_argument("--method", type=str, default="wl")
parser.add_argument("--mode", type=str, default="none")
add_bool_arg(parser, "debug")
args = parser.parse_args()


print("Processing:\t", f"graph{args.node}{args.ac}.g6")
with open(f"g6/graph{args.node}{args.ac}.g6", "r") as f:
    graphkac = [x.strip() for x in f.readlines()]


hash_set = set()
hash_dict = dict()
hash_dict_repeat = dict()
for (id, x) in tqdm(enumerate(graphkac)):
    graph = nx.from_graph6_bytes(x.encode())
    if args.method == "wl":
        graph_hash = wl_hash_tqdm(G=graph, k=args.wl, debug=args.debug)
    elif args.method == "fwl":
        graph_hash = fwl_hash_tqdm(G=graph, k=args.wl, debug=args.debug)
    elif args.method == "1wl":
        graph_hash = wl_1_hash_tqdm(G=graph, mode=args.mode, debug=args.debug)
    elif args.method == "nx":
        graph_hash = nx.weisfeiler_lehman_graph_hash(G=graph, iterations=args.node)
    else:
        print("No such method")
        exit()
    hash_dict[x] = graph_hash
    if graph_hash not in hash_set:
        hash_set.add(graph_hash)
    else:
        if graph_hash not in hash_dict_repeat:
            hash_dict_repeat[graph_hash] = 2
        else:
            hash_dict_repeat[graph_hash] += 1


if args.method == "1wl":
    print("Saving:\t", f"Kac/raw/{args.node}{args.ac}_1wl_mode_{args.mode}.npy")
    np.save(f"Kac/raw/{args.node}{args.ac}_1wl_mode_{args.mode}.npy", hash_dict)
elif args.method == "wl":
    print("Saving:\t", f"Kac/raw/{args.node}{args.ac}_{args.wl}wl.npy")
    np.save(f"Kac/raw/{args.node}{args.ac}_{args.wl}wl.npy", hash_dict)
elif args.method == "fwl":
    print("Saving:\t", f"Kac/raw/{args.node}{args.ac}_{args.wl}fwl.npy")
    np.save(f"Kac/raw/{args.node}{args.ac}_{args.wl}fwl.npy", hash_dict)
else:
    print("Saving:\t", f"Kac/raw/{args.node}{args.ac}_nxwl.npy")
    np.save(f"Kac/raw/{args.node}{args.ac}_nxwl.npy", hash_dict)

res = 0
for (key, value) in hash_dict_repeat.items():
    res += int(value * (value - 1) / 2)
print(res)
