import numpy as np
import scipy as sp
import scipy.sparse
import networkx as nx
import itertools
import pickle as pkl
from argparse import ArgumentParser
from tqdm import tqdm
from gf.utils import *


def gen_graphs(V_min = 50, V_max = 400):
    G = pkl.load(open(f"datasets/ego/ind.citeseer.graph", "rb"), 
                      encoding="latin1")
    G = nx.from_dict_of_lists(G)
    G = max(nx.connected_component_subgraphs(G), key=len)
    G = nx.convert_node_labels_to_integers(G)
    A = []
    for i in tqdm(range(G.number_of_nodes())):
        G_ego = nx.ego_graph(G, i, radius = 3)
        if (G_ego.number_of_nodes() >= V_min and \
            G_ego.number_of_nodes() <= V_max):
            A.append(nx.to_numpy_array(G_ego))
    V = [len(a) for a in A]
    return np.array(V), np.array(A)


if __name__ == "__main__":
    
    argparser = ArgumentParser()
    argparser.add_argument("--train-N", default=600, type=int)
    argparser.add_argument("--seed", default=123, type=int)
    args = argparser.parse_args()

    np.random.seed(args.seed)

    V, A = gen_graphs()
    E = np.array([compute_fgsd_embeddings(a) for a in A])

    train_idxs = np.random.choice(len(V), args.train_N, replace = False)
    test_idxs = np.setdiff1d(np.arange(len(V)), train_idxs)

    np.save("datasets/ego/train_A.npy", A[train_idxs])
    np.save("datasets/ego/train_E.npy", E[train_idxs])
    np.save("datasets/ego/train_V.npy", V[train_idxs])

    np.save("datasets/ego/test_A.npy", A[test_idxs])
    np.save("datasets/ego/test_E.npy", E[test_idxs])
    np.save("datasets/ego/test_V.npy", V[test_idxs])

