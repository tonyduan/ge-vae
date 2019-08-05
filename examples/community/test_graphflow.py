import logging
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networkx as nx
import matplotlib as mpl
from argparse import ArgumentParser
from graphflows.graphflow import GF, EdgePredictor
from graphflows.fgsd import compute_fgsd_embeddings
from tqdm import tqdm
mpl.use("agg")
from matplotlib import pyplot as plt


def gen_graphs(sizes, p_intra=0.7, p_inter=0.01):
    """
    Generate community graphs.
    """
    max_size = np.max(sizes)
    X = np.zeros((len(sizes), max_size, 32))
    A = np.zeros((len(sizes), max_size, max_size))
    for idx, V in enumerate(sizes):
        comms = [nx.gnp_random_graph(V // 2, p_intra),
                 nx.gnp_random_graph((V + 1) // 2, p_intra)]
        graph = nx.disjoint_union_all(comms)
        graph = nx.to_numpy_array(graph)
        block1 = np.arange(V // 2)
        block2 = np.arange(V // 2, V)
        remaining = list(itertools.product(block1, block2))
        np.random.shuffle(remaining)
        for (i, j) in remaining[:int(p_inter * V + 1)]:
            graph[i,j] = 1
            graph[j,i] = 1
        P = np.eye(V)
        np.random.shuffle(P)
        graph = P.T @ graph @ P
        graph = np.pad(graph, (0, max_size - V), "constant")
        X[idx,:] = np.r_[0.5 * np.random.randn(V, 32),
                         np.zeros((max_size - V, 32))]
        A[idx] = graph
    return X, A


def convert_pairwise(A, E):
    """
    Convert to a representation with a pairwise relationship between each node.

    First and second row represent embeddings for the pair in question.
    Rest of the rows represent embeddings for all remaining pairs.
    """
    X, Y, idxs = [], [], []
    for A_k, E_k in tqdm(zip(A, E), total=len(A)):
        for (i, j) in itertools.combinations(np.arange(len(A_k)), 2):
            Y += [A_k[i, j]]
            first = E_k[i][np.newaxis,:]
            second = E_k[j][np.newaxis,:]
            rest_idx = np.r_[np.arange(i), np.arange(i + 1, j),
                             np.arange(j + 1, len(A_k))]
            rest = np.take(E_k, rest_idx, axis=0)
            X += [np.r_[first, second, rest]]
            idxs+= [(i, j)]
    return idxs, np.array(X), np.array(Y)


def construct_pairwise_X(E):
    X, idxs = [], []
    for E_k in tqdm(E, total=len(E)):
        for (i, j) in itertools.combinations(np.arange(len(E_k)), 2):
            first = E_k[i][np.newaxis,:]
            second = E_k[j][np.newaxis,:]
            rest_idx = np.r_[np.arange(i), np.arange(i + 1, j),
                             np.arange(j + 1, len(E_k))]
            rest = np.take(E_k, rest_idx, axis=0)
            X += [np.r_[first, second, rest]]
            idxs += [(i, j)]
    return idxs, np.array(X)

def reconstruct_adjacency(N, idxs, Y_hat):
    """
    Reconstruct the adjacency matrix from a set of indices and predictions.
    """
    A = np.zeros((N, N))
    for (i, j), y in zip(idxs, Y_hat):
        A[i,j] = y
        A[j,i] = y
    return A

def plot_graphs(X, A):
    plt.figure(figsize=(8, 3))
    for idx in range(len(A)):
        plt.subplot(1, len(A), idx + 1)
        feature_norms = np.linalg.norm(X[idx], axis = 1)
        graph = A[idx][(feature_norms > 0),:][:,(feature_norms > 0)]
        G = nx.from_numpy_matrix(graph)
        nx.draw(G, node_color = "black", node_size = 50)
    plt.show()



if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--N", default=1000, type=int)
    argparser.add_argument("--K", default=4, type=int)
    argparser.add_argument("--edgepredictor-file", default="ep.torch")
    argparser.add_argument("--graphflow-file", default="gf.torch")
    args = argparser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    sizes = np.random.choice(np.arange(18, 19), size=args.N)

    edge_predictor = EdgePredictor(args.K)
    edge_predictor.load_state_dict(torch.load(f"./ckpts/{args.edgepredictor_file}"))

    model = GF(n_nodes = 18, embedding_dim = args.K, num_flows = 2, device = "cpu")
    model.load_state_dict(torch.load(f"./ckpts/{args.graphflow_file}"))

    plt.figure(figsize=(8, 5))
    losses = np.load("./ckpts/loss_curve.npy")
    plt.plot(np.arange(len(losses)) + 1, losses, color = "black")
    plt.title("Training loss")
    plt.savefig("./ckpts/img/loss.png")

    plt.figure(figsize=(8, 6))
    for i in range(8):

        X, A = gen_graphs([18])
        E_true = np.array([compute_fgsd_embeddings(a) for a in A])
        E_true = E_true[:, :, :args.K]
        idxs, X, Y = convert_pairwise(A, E_true)
        X = torch.tensor(X, dtype=torch.float)
        Y_hat = torch.sigmoid(edge_predictor.forward(X))
        A_hat = reconstruct_adjacency(18, idxs, Y_hat)
        A_sample = (np.random.rand(*A_hat.shape) < A_hat).astype(int)
        plt.subplot(4, 2, i + 1)
        nx.draw(nx.from_numpy_array(A_sample), node_color = "black",
                node_size = 50)
    plt.show()
    plt.savefig("./ckpts/img/ep.png")

    X, A = gen_graphs(sizes)
    E_true = np.array([compute_fgsd_embeddings(a) for a in A])
    E_true = E_true[:, :, :args.K]

    Z_true, _, _ = model.forward(torch.tensor(E_true, dtype=torch.float))
    Z_true = Z_true.data.numpy()
    plt.figure(figsize=(8, 6))
    for i in range(Z_true.shape[2]):
        plt.subplot(2, 2, i + 1)
        plt.hist(Z_true[:, :, i], bins = 30)
    plt.tight_layout()
    plt.savefig("./ckpts/img/hists.png")

    plt.figure(figsize=(10, 12))
    for i in range(4):

        plt.subplot(4, 4, i + 1)
        plt.imshow(E_true[i], vmin = -0.8, vmax = 0.8)
        plt.title("Truth")

        Z_true, _, _ = model.forward(torch.tensor(E_true, dtype=torch.float))

        plt.subplot(4, 4, i + 5)
        plt.imshow(Z_true[i].data.numpy(), cmap = "inferno")
        plt.title("Corresponding prior")

        Z = model.sample_prior(1)
        E = model.backward(Z)
        sigma = torch.load("./ckpts/sigma.torch").cpu()
        mu = torch.load("./ckpts/mu.torch").cpu()
        E = E * sigma + mu

        plt.subplot(4, 4, i + 9)
        plt.imshow(E[0].data.numpy(), vmin = -0.8, vmax = 0.8)
        plt.title("Generated")

        idxs, X = construct_pairwise_X(E.data.numpy())
        X = torch.tensor(X, dtype=torch.float)
        Y_hat = torch.sigmoid(edge_predictor.forward(X))
        A_hat = reconstruct_adjacency(18, idxs, Y_hat)

        plt.subplot(4, 4, i + 13)
        A_sample = (np.random.rand(*A_hat.shape) < A_hat).astype(int)
        nx.draw(nx.from_numpy_array(A_sample), node_color = "black",
                node_size = 50)
        plt.title("Generated")

    plt.tight_layout()
    plt.show()
    plt.savefig("./ckpts/img/gf.png")