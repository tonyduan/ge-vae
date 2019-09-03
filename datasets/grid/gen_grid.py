
import numpy as np
import networkx as nx
import itertools
from argparse import ArgumentParser
from gf.utils import *
from tqdm import tqdm


def gen_graphs(sizes, p_intra=0.7, p_inter=0.01):
    """
    Generate grid graphs.
    """
    A = []
    for V in tqdm(sizes):
        G = nx.grid_2d_graph(int(round(V ** 0.5)), int(round(V ** 0.5)))
        G = nx.to_numpy_array(G)
        # due to symmetries in grid dataset permutations cause issues
        # this arises from numerical imprecision in eigenvector calculation
        #P = np.eye(int(round(V ** 0.5)) ** 2)  
        #np.random.shuffle(P)
        #A.append(P.T @ G @ P)
        A.append(G)
    return np.array(A)


if __name__ == "__main__":
    
    argparser = ArgumentParser()
    argparser.add_argument("--train-N", default=2500, type=int)
    argparser.add_argument("--test-N", default=1000, type=int)
    argparser.add_argument("--seed", default=123, type=int)
    args = argparser.parse_args()

    np.random.seed(args.seed)

    V = np.random.choice(np.arange(9, 25), size = args.train_N + args.test_N)
    A = gen_graphs(V)
    V = np.array([len(a) for a in A])
    E = np.array([compute_fgsd_embeddings(a) for a in A])

    K = min([e.shape[1] for e in E])
    E = np.array([e[:, :K] for e in E])

    train_idxs = np.random.choice(len(V), args.train_N, replace = False)
    test_idxs = np.setdiff1d(np.arange(len(V)), train_idxs)

    np.save("datasets/grid/train_A.npy", A[train_idxs])
    np.save("datasets/grid/train_E.npy", E[train_idxs])
    np.save("datasets/grid/train_V.npy", V[train_idxs])

    np.save("datasets/grid/test_A.npy", A[test_idxs])
    np.save("datasets/grid/test_E.npy", E[test_idxs])
    np.save("datasets/grid/test_V.npy", V[test_idxs])
