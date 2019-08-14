import logging
import itertools
import numpy as np
import torch
import numpy as np
import networkx as nx
import matplotlib as mpl
from argparse import ArgumentParser
from gf.models.gfvae import GFVAE
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


def plot_prior_histograms(model, E, A, V):
    Z = []
    for i in tqdm(range(len(E))):
        z, _, _ = model.forward(torch.tensor([E[i]], dtype = torch.float), 
                                torch.tensor([A[i]], dtype = torch.float),
                                torch.tensor([V[i]], dtype = torch.float))
        Z.append(z[0].data.numpy())
    Z = np.vstack(Z)
    plt.figure(figsize=(8, 6))
    for i in range(Z.shape[1]):
        plt.subplot(2, 2, i + 1)
        plt.hist(Z[:, i], bins = 30)
    plt.tight_layout()
    plt.savefig("./ckpts/img/hists.png")

def interpolate(model, x1, x2, x3, x4, mu, sigma):
    n_nodes = x1.shape[0]
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


def show_embeddings_and_samples(model, E, A, V, mu, sigma):
    plt.figure(figsize=(10, 12))
    for i in range(4):

        plt.subplot(4, 4, i + 1)
        plt.imshow(E[i], vmin = -0.8, vmax = 0.8)
        plt.title("Truth")

        Z = model.forward(torch.tensor([E[i]], dtype = torch.float), 
                          torch.tensor([A[i]], dtype = torch.float),
                          torch.tensor([V[i]], dtype = torch.float))[0][0]

        plt.subplot(4, 4, i + 5)
        plt.imshow(Z.data.numpy(), cmap = "inferno")
        plt.title("Corresponding prior")

        Z_sampled, V_sampled = model.sample_prior(1)
        E_sampled = model.backward(Z_sampled, V_sampled)[0].data.numpy()

        plt.subplot(4, 4, i + 9)
        plt.imshow(E_sampled, vmin = -0.8, vmax = 0.8)
        plt.title("Generated")

        A_hat = torch.sigmoid(model.ep.forward(torch.tensor([E_sampled], dtype=torch.float),
                                               torch.tensor([V_sampled], dtype=torch.float)))[0].data.numpy()
        Z = np.random.rand(*A_hat.shape)
        A_sample = (np.tril(Z) + np.tril(Z, -1).T < A_hat).astype(int)

        plt.subplot(4, 4, i + 13)
        nx.draw(nx.from_numpy_array(A_sample), node_color = "black",
                node_size = 20)
        plt.title("Generated")

    plt.tight_layout()
    plt.show()
    plt.savefig("./ckpts/img/gf.png")


def plot_ep_samples(model, E, A, V):
    plt.figure(figsize=(8, 3))
    for i in range(len(E)):
        e = torch.tensor([E[i]], dtype = torch.float)
        v = torch.tensor([V[i]], dtype = torch.float)
        a = torch.tensor([A[i]], dtype = torch.float)
        A_hat = model.predict_A_from_E(e, a, v).data.numpy()[0]
        Z = np.random.rand(*A_hat.shape)
        A_sample = (np.tril(Z) + np.tril(Z, -1).T < A_hat).astype(int)
        plt.subplot(2, 4, i + 1)
        nx.draw(nx.from_numpy_array(A_sample), node_color = "black",
                node_size = 20)
    plt.tight_layout()
    plt.savefig("./ckpts/img/ep.png")


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--dataset", default="community")
    argparser.add_argument("--K", default=4, type=int)
    args = argparser.parse_args()

    E = np.load(f"datasets/{args.dataset}/test_E.npy", allow_pickle = True)
    A = np.load(f"datasets/{args.dataset}/test_A.npy", allow_pickle = True)
    V = np.load(f"datasets/{args.dataset}/test_V.npy")
    E = [e[:, :args.K] for e in E]

    model = GFVAE(embedding_dim = args.K, num_mp_steps = 5, device = "cpu")
    model.load_state_dict(torch.load("./ckpts/gfvae/weights.torch"))
    mu = np.load("./ckpts/gfvae/mu.npy")
    sigma = np.load("./ckpts/gfvae/sigma.npy")
    E = [(e - mu) / sigma for e in E]

    # == create the figures
    plot_sample_graphs(A[:4])
    #plot_prior_histograms(model, E, A, V)
    #show_embeddings_and_samples(model, E[:4], A[:4], V[:4], mu, sigma)
    idx = np.random.choice(len(E) - 8)
    plot_ep_samples(model, E[idx:idx + 8], A[idx:idx+8], V[idx:idx+8])
    #E = list(filter(lambda e: len(e) == 16, E))
    #interpolate(model, edge_predictor, E[0], E[1], E[2], E[3], mu, sigma)

