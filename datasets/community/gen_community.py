import numpy as np
import networkx as nx
import itertools
from argparse import ArgumentParser
from gf.utils import *
from tqdm import tqdm


def gen_graphs(sizes, p_intra=0.7, p_inter=0.01):
    """
    Generate community graphs.
    """
    A = []
    for V in tqdm(sizes):
        comms = [nx.gnp_random_graph(V // 2, p_intra),
                 nx.gnp_random_graph((V + 1) // 2, p_intra)]
        graph = nx.disjoint_union_all(comms)
        graph = nx.to_numpy_array(graph)
        block1 = np.arange(V // 2)
        block2 = np.arange(V // 2, V)
        remaining = list(itertools.product(block1, block2))
        np.random.shuffle(remaining)
        for (i, j) in remaining[:int(p_inter * V + 1)]:
            graph[i,j], graph[j,i] = 1, 1
        P = np.eye(V)
        np.random.shuffle(P)
        graph = P.T @ graph @ P
        if nx.number_connected_components(nx.from_numpy_array(graph)) > 1:
            sizes = sizes + [V]
            continue
        A.append(graph)
    return np.array(A)


if __name__ == "__main__":
    
    argparser = ArgumentParser()
    argparser.add_argument("--train-N", default=2500, type=int)
    argparser.add_argument("--test-N", default=1000, type=int)
    argparser.add_argument("--seed", default=123, type=int)
    args = argparser.parse_args()

    np.random.seed(args.seed)

    V = np.random.choice(np.arange(19, 20), size = args.train_N + args.test_N)
    A = gen_graphs(V)
    V = np.array([len(a) for a in A])
    E = np.array([compute_fgsd_embeddings(a) for a in A])

    K = min([e.shape[1] for e in E])
    E = np.array([e[:, :K] for e in E])

    train_idxs = np.random.choice(len(V), args.train_N, replace = False)
    test_idxs = np.setdiff1d(np.arange(len(V)), train_idxs)

    np.save("datasets/community/train_A.npy", A[train_idxs])
    np.save("datasets/community/train_E.npy", E[train_idxs])
    np.save("datasets/community/train_V.npy", V[train_idxs])

    np.save("datasets/community/test_A.npy", A[test_idxs])
    np.save("datasets/community/test_E.npy", E[test_idxs])
    np.save("datasets/community/test_V.npy", V[test_idxs])
