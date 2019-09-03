
import numpy as np
import networkx as nx
import itertools
from argparse import ArgumentParser
from gf.utils import *
from gf.embeddings import *
from tqdm import tqdm


def gen_graphs(widths, heights, p_intra=0.7, p_inter=0.01):
    """
    Generate grid graphs.
    """
    A = []
    for w, h in tqdm(zip(widths, heights), total = len(widths)):
        G = nx.grid_2d_graph(w, h)
        G = nx.to_numpy_array(G)
        P = np.eye(w * h)
        np.random.shuffle(P)
        A.append(P.T @ G @ P)
    return np.array(A)


if __name__ == "__main__":
    
    argparser = ArgumentParser()
    argparser.add_argument("--train-N", default=2500, type=int)
    argparser.add_argument("--test-N", default=1000, type=int)
    argparser.add_argument("--seed", default=123, type=int)
    args = argparser.parse_args()

    np.random.seed(args.seed)

    W = np.random.choice(np.arange(10, 20), size = args.train_N + args.test_N)
    H = np.random.choice(np.arange(10, 20), size = args.train_N + args.test_N)
    A = gen_graphs(W, H)
    V = np.array([len(a) for a in A])
    E = np.array([compute_fgsd_embeddings(a) for a in A])

    K = min([e.shape[1] for e in E])
    E = np.array([e[:, :K] for e in E])

    train_idxs = np.random.choice(len(V), args.train_N, replace = False)
    test_idxs = np.setdiff1d(np.arange(len(V)), train_idxs)

    np.save("datasets/grid_big/train_A.npy", A[train_idxs])
    np.save("datasets/grid_big/train_E.npy", E[train_idxs])
    np.save("datasets/grid_big/train_V.npy", V[train_idxs])

    np.save("datasets/grid_big/test_A.npy", A[test_idxs])
    np.save("datasets/grid_big/test_E.npy", E[test_idxs])
    np.save("datasets/grid_big/test_V.npy", V[test_idxs])

