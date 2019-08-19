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
        G = nx.barabasi_albert_graph(V, 3)
        G = nx.to_numpy_array(G)
        P = np.eye(V)
        np.random.shuffle(P)
        A.append(P.T @ G @ P)
    return np.array(A)


if __name__ == "__main__":
    
    argparser = ArgumentParser()
    argparser.add_argument("--train-N", default=2500, type=int)
    argparser.add_argument("--test-N", default=1000, type=int)
    argparser.add_argument("--seed", default=123, type=int)
    argparser.add_argument("--noise", default=0.025, type=float)
    args = argparser.parse_args()

    np.random.seed(args.seed)

    V = np.random.choice(np.arange(12, 20), size = args.train_N + args.test_N)
    A = gen_graphs(V)
    V = np.array([len(a) for a in A])
    E = np.array([compute_fgsd_embeddings(a) for a in A])

    E = [e + args.noise * np.random.randn(*e.shape) for e in E]
    K = min([e.shape[1] for e in E])
    E = [e[:, :K] for e in E]
    mu = np.mean(np.vstack(E), axis = 0)
    sigma = np.std(np.vstack(E), axis = 0)
    E = np.array([(e - mu) / sigma for e in E])

    train_idxs = np.random.choice(len(V), args.train_N, replace = False)
    test_idxs = np.setdiff1d(np.arange(len(V)), train_idxs)

    np.save("datasets/ba/train_A.npy", A[train_idxs])
    np.save("datasets/ba/train_E.npy", E[train_idxs])
    np.save("datasets/ba/train_V.npy", V[train_idxs])

    np.save("datasets/ba/test_A.npy", A[test_idxs])
    np.save("datasets/ba/test_E.npy", E[test_idxs])
    np.save("datasets/ba/test_V.npy", V[test_idxs])

    np.save("datasets/ba/mu.npy", mu)
    np.save("datasets/ba/sigma.npy", sigma)
