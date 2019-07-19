import logging
import itertools
import torch
import torch.optim as optim
import numpy as np
import networkx as nx
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from graphflows.gnf import GRevNet, GAE
from graphflows.fgsd import compute_fgsd_embeddings


def sort_A(A):
    return np.array(sorted(A, key=lambda r: np.sum(r), reverse=True))

def gen_graphs(sizes, p_intra=0.7, p_inter=0.01):
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
    breakpoint()
    return X, A


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
    argparser.add_argument("--N", default=200, type=int)
    argparser.add_argument("--iterations", default=20, type=int)
    argparser.add_argument("--train", action="store_true")
    args = argparser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    sizes = np.random.choice(np.arange(18, 19), size=args.N)
    X, A = gen_graphs(sizes)
    E = np.array([compute_fgsd_embeddings(a) for a in A])

    A_fake = np.ones((200, 18, 18))
    ind1, ind2 = np.diag_indices(18)
    for i in range(200):
        A_fake[i][ind1, ind2] = 0
    A_fake = torch.tensor(A_fake, dtype=torch.float)

    X = torch.tensor(X, dtype=torch.float)
    A = torch.tensor(A, dtype=torch.float)
    E = torch.tensor(E, dtype=torch.float)

    model = GRevNet(hidden_dim = 18, message_dim = 16, num_layers = 4)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for i in range(args.iterations):
        optimizer.zero_grad()
        Z, prior_logprob, log_det = model.forward(E, A_fake)
        loss = -torch.mean(prior_logprob + log_det)
        loss.backward()
        optimizer.step()
        if i % 1 == 0:
            logger.info(f"Iter: {i}\t" +
                        f"Logprob: {(prior_logprob.mean() + log_det.mean()).data:.2f}\t" +
                        f"Prior: {prior_logprob.mean().data:.2f}\t" +
                        f"LogDet: {log_det.mean().data:.2f}")

    breakpoint()
    Z = torch.randn(1, 18, 18)
    X = model.backward(Z, A_fake[0])
    plt.subplot(2, 2, 1)
    plt.imshow(model.flows[0].F1.mpl.attn_weights.data[0] + model.flows[0].G1.mpl.attn_weights.data[0] + \
               model.flows[0].F2.mpl.attn_weights.data[0] + model.flows[0].G2.mpl.attn_weights.data[0])
    plt.subplot(2, 2, 2)
    plt.imshow(model.flows[1].F1.mpl.attn_weights.data[0] + model.flows[1].G1.mpl.attn_weights.data[0] + \
               model.flows[1].F2.mpl.attn_weights.data[0] + model.flows[1].G2.mpl.attn_weights.data[0])
    plt.subplot(2, 2, 3)
    plt.imshow(model.flows[2].F1.mpl.attn_weights.data[0] + model.flows[2].G1.mpl.attn_weights.data[0] + \
               model.flows[2].F2.mpl.attn_weights.data[0] + model.flows[2].G2.mpl.attn_weights.data[0])
    plt.subplot(2, 2, 4)
    plt.imshow(model.flows[3].F1.mpl.attn_weights.data[0] + model.flows[3].G1.mpl.attn_weights.data[0] + \
               model.flows[3].F2.mpl.attn_weights.data[0] + model.flows[3].G2.mpl.attn_weights.data[0])
    plt.show()

    W = model.flows[3].F1.mpl.attn_weights.data[0] + model.flows[3].G1.mpl.attn_weights.data[0] + \
                   model.flows[3].F2.mpl.attn_weights.data[0] + model.flows[3].G2.mpl.attn_weights.data[0]