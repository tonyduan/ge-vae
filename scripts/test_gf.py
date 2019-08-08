import logging
import itertools
import numpy as np
import torch
import numpy as np
import networkx as nx
import matplotlib as mpl
from argparse import ArgumentParser
from gf.models.graphflow import GF
from gf.models.ep import EdgePredictor
from gf.utils import *
from tqdm import tqdm
from matplotlib import pyplot as plt
mpl.use("agg")



def plot_sample_graphs(A):
    plt.figure(figsize=(8, 3))
    for idx in range(len(A)):
        plt.subplot(1, len(A), idx + 1)
        graph = A[idx]
        G = nx.from_numpy_matrix(graph)
        nx.draw(G, node_color = "black", node_size = 20)
    plt.savefig("./ckpts/img/sample.png")


def plot_prior_histograms(model, E):
    Z = []
    for e in tqdm(E):
        z, _ = model.forward(torch.tensor([e], dtype=torch.float))
        Z.append(z[0].data.numpy())
    Z = np.vstack(Z)
    plt.figure(figsize=(8, 6))
    for i in range(Z.shape[1]):
        plt.subplot(2, 2, i + 1)
        plt.hist(Z[:, i], bins = 30)
    plt.tight_layout()
    plt.savefig("./ckpts/img/hists.png")

def interpolate(model, edgepredictor, x1, x2, x3, x4, mu, sigma):
    n_nodes = x1.shape[0]
    x1 = torch.tensor([(x1 - mu) / sigma], dtype = torch.float)
    x2 = torch.tensor([(x2 - mu) / sigma], dtype = torch.float)
    x3 = torch.tensor([(x3 - mu) / sigma], dtype = torch.float)
    x4 = torch.tensor([(x4 - mu) / sigma], dtype = torch.float)
    z1 = model.forward(x1)[0].data.numpy()[0]
    z2 = model.forward(x2)[0].data.numpy()[0]
    z3 = model.forward(x3)[0].data.numpy()[0]
    z4 = model.forward(x4)[0].data.numpy()[0]
    x = np.arange(0, 8 * np.pi / 14, np.pi / 14)
    y = np.arange(0, 8 * np.pi / 14, np.pi / 14)
    plt.figure(figsize=(16, 16))
    for i in range(64):
        plt.subplot(8, 8, i + 1)
        x = i % 8 * np.pi / 14
        y = i // 8 * np.pi / 14
        z = np.cos(x) * (np.cos(y) * z1 + np.sin(y) * z2) + \
            np.sin(x) * (np.cos(y) * z3 + np.sin(y) * z4)
        z = torch.tensor(z, dtype=torch.float)
        E = model.backward(z.unsqueeze(0))[0].data.numpy()
        E = E * sigma + mu
        idxs, X = convert_embeddings_pairwise(E[np.newaxis,:])
        X = torch.tensor(X, dtype=torch.float)
        Y = torch.sigmoid(edge_predictor.forward(X))
        A = reconstruct_adjacency_matrix(n_nodes, idxs, Y)
        A = (np.random.rand(*A.shape) < A).astype(int)
        nx.draw(nx.from_numpy_array(A), node_color="black",
                node_size = 20)
    plt.tight_layout()
    plt.savefig("./ckpts/img/interpolate.png")


def show_embeddings_and_samples(model, edgepredictor, E, mu, sigma):
    plt.figure(figsize=(10, 12))
    for i in range(4):

        plt.subplot(4, 4, i + 1)
        plt.imshow(E[i], vmin = -0.8, vmax = 0.8)
        plt.title("Truth")

        E_std = (E[i] - mu) / sigma
        Z = model.forward(torch.tensor([E_std], dtype=torch.float))[0][0]

        plt.subplot(4, 4, i + 5)
        plt.imshow(Z.data.numpy(), cmap = "inferno")
        plt.title("Corresponding prior")

        Z_sampled = model.sample_prior(1)
        E_sampled = model.backward(Z_sampled)[0].data.numpy()
        E_sampled = E_sampled * sigma + mu

        plt.subplot(4, 4, i + 9)
        plt.imshow(E_sampled, vmin = -0.8, vmax = 0.8)
        plt.title("Generated")

        idxs, X = convert_embeddings_pairwise([E_sampled])
        #idxs, X = convert_embeddings_pairwise([E[i]])
        X = torch.tensor(X, dtype=torch.float)
        Y_hat = torch.sigmoid(edge_predictor.forward(X))
        A_hat = reconstruct_adjacency_matrix(X.shape[1], idxs, Y_hat)

        plt.subplot(4, 4, i + 13)
        A_sample = (np.random.rand(*A_hat.shape) < A_hat).astype(int)
        nx.draw(nx.from_numpy_array(A_sample), node_color = "black",
                node_size = 20)
        plt.title("Generated")

    plt.tight_layout()
    plt.show()
    plt.savefig("./ckpts/img/gf.png")


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--dataset", default="community")
    argparser.add_argument("--K", default=4, type=int)
    args = argparser.parse_args()

    E = np.load(f"datasets/{args.dataset}/test_E.npy", allow_pickle = True)
    A = np.load(f"datasets/{args.dataset}/test_A.npy", allow_pickle = True)
    V = np.load(f"datasets/{args.dataset}/test_V.npy")
    E = [e[:, :args.K] for e in E]

    # == load the relevant models
    edge_predictor = EdgePredictor(args.K)
    edge_predictor.load_state_dict(torch.load("./ckpts/ep/weights.torch"))

    model = GF(embedding_dim = args.K, num_flows = 2, device = "cpu")
    model.load_state_dict(torch.load("./ckpts/gf/weights.torch"))
    mu = np.load("./ckpts/gf/mu.npy")
    sigma = np.load("./ckpts/gf/sigma.npy")

    # == create the figures
    plot_sample_graphs(A[:4])
    plot_prior_histograms(model, E)
    show_embeddings_and_samples(model, edge_predictor, E[:4], mu, sigma)
    E = list(filter(lambda e: len(e) == 10, E))
    interpolate(model, edge_predictor, E[0], E[1], E[2], E[3], mu, sigma)
