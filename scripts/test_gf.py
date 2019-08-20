import numpy as np
import torch
import json
import networkx as nx
import matplotlib as mpl
import scipy as sp
import scipy.stats
from argparse import ArgumentParser
from gf.models.gf import GF
from gf.models.ep import EdgePredictor
from gf.datasets import *
from gf.utils import *
from gf.eval.stats import *
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from matplotlib import pyplot as plt


def get_largest_cc(A):
    cc = list(max(nx.connected_components(nx.from_numpy_array(A)), key = len))
    cc = np.array(cc)
    A = A[cc, :][:, cc]
    A[np.arange(len(A)), np.arange(len(A))] = 0
    return A

def plot_sample_graphs(model, dataloader):
    _, A, _ = next(iter(dataloader))
    n_graphs = len(A)
    plt.figure(figsize=(n_graphs, 4))
    for idx in range(n_graphs):
        plt.subplot(2, n_graphs // 2, idx + 1)
        G = get_largest_cc(A[idx].data.numpy())
        G = nx.from_numpy_matrix(G)
        nx.draw(G, node_color = "black", node_size = 20)
    plt.tight_layout()


def plot_prior_histograms(model, dataloader):
    z = []
    for E_batch, A_batch, V_batch in tqdm(iter(dataloader)):
        z_batch, _, _ = model.forward(E_batch, A_batch, V_batch)
        z += [np.vstack(z_batch.data.numpy())]
    z = np.vstack(z)
    x_axis = np.linspace(-4, 4, 200)
    embedding_dim = z.shape[-1]
    plt.figure(figsize = (embedding_dim, 4))
    for i in range(embedding_dim):
        plt.subplot(2, embedding_dim // 2, i  + 1)
        density = sp.stats.gaussian_kde(z[:, i], "silverman")
        plt.plot(x_axis, density(x_axis))
        plt.fill_between(x_axis, 0, density(x_axis), alpha=0.2)
        plt.title(f"Dimension {i + 1}")
    plt.tight_layout()


def plot_interpolations(model, dataloader):
    x_batch, a_batch, v_batch = next(iter(dataloader))
    n_nodes = v_batch[0].unsqueeze(0)
    z_batch, _, _ = model.forward(x_batch, a_batch, v_batch)
    x = np.arange(0, 8 * np.pi / 14, np.pi / 14)
    y = np.arange(0, 8 * np.pi / 14, np.pi / 14)
    plt.figure(figsize=(16, 16))
    for i in tqdm(range(64)):
        plt.subplot(8, 8, i + 1)
        x = i % 8 * np.pi / 14
        y = i // 8 * np.pi / 14
        z = np.cos(x) * (np.cos(y) * z_batch[0] + np.sin(y) * z_batch[1]) + \
            np.sin(x) * (np.cos(y) * z_batch[2] + np.sin(y) * z_batch[3])
        e = model.backward(z.unsqueeze(0), n_nodes)
        a_hat = model.ep.forward(e, n_nodes)[0]
        a_hat = torch.sigmoid(a_hat).data.numpy()
        z = np.random.rand(*a_hat.shape)
        a_sample = (np.tril(z) + np.tril(z, -1).T < a_hat).astype(int)
        cc = get_largest_cc(a_sample)
        nx.draw(nx.from_numpy_array(cc), node_color = "black", node_size = 20)
    plt.tight_layout()


def plot_generations(n_batch, n_nodes):
    z_sampled, v_sampled = model.sample_prior(n_batch, n_nodes)
    e_sampled = model.backward(z_sampled, v_sampled)
    a_hat = torch.sigmoid(model.ep.forward(e_sampled, v_sampled)).data.numpy()
    plt.figure(figsize = (n_batch, 4))
    for i in range(n_batch):
        z = np.random.rand(*a_hat[i].shape)
        a_sample = (np.tril(z) + np.tril(z, -1).T < a_hat[i]).astype(int)
        cc = get_largest_cc(a_sample)
        plt.subplot(2, n_batch // 2, i + 1)
        nx.draw(nx.from_numpy_array(cc), node_color = "black", node_size = 20)
    plt.tight_layout()


def plot_reconstructions(model, dataloader):
    x_batch, _, v_batch = next(iter(dataloader))
    a_hat = model.predict_a_from_e(x_batch, v_batch).data.numpy()
    n_graphs = len(x_batch)
    plt.figure(figsize=(n_graphs, 4))
    for i in range(n_graphs):
        z = np.random.rand(*a_hat[i].shape)
        a_sample = (np.tril(z) + np.tril(z, -1).T < a_hat[i]).astype(int)
        cc = get_largest_cc(a_sample)
        plt.subplot(2, n_graphs // 2, i + 1)
        nx.draw(nx.from_numpy_array(cc), node_color = "black", node_size = 20)
    plt.tight_layout()


def compute_test_bpd(model, dataloader):
    """
    Compute the test BPD per embedding dimension and per edge.
    """
    node_bpds, edge_bpds = [], []
    for E_batch, A_batch, V_batch in tqdm(iter(dataloader)):
        _, node_lp, edge_lp = model.forward(E_batch, A_batch, V_batch)
        node_bpds += [-node_lp.data.numpy() / np.log(2)]
        edge_bpds += [-edge_lp.data.numpy() / np.log(2)]
    return np.concatenate(node_bpds), np.concatenate(edge_bpds)


def generate_for_test_set(model, dataloader, batch_size = 128):
    gen_graphs = []
    for _, _, v in tqdm(iter(dataloader)):
        z_sampled, _ = model.sample_prior(len(v), int(max(v)))
        e_sampled = model.backward(z_sampled, v)
        mask = construct_adjacency_mask(v)
        a_hat = torch.sigmoid(model.ep.forward(e_sampled, v)) * mask
        a_hat = a_hat.data.numpy()
        for i in range(len(a_hat)):
            z = np.random.rand(*a_hat[i].shape)
            a_sample = (np.tril(z) + np.tril(z, -1).T < a_hat[i]).astype(int)
            cc = get_largest_cc(a_sample)
            gen_graphs.append(cc)
    return np.array(gen_graphs)


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--dataset", default="community")
    argparser.add_argument("--K", default=8, type=int)
    argparser.add_argument("--batch-size", default=128, type=int)
    args = argparser.parse_args()

    E = np.load(f"datasets/{args.dataset}/test_E.npy", allow_pickle = True)
    A = np.load(f"datasets/{args.dataset}/test_A.npy", allow_pickle = True)
    V = np.load(f"datasets/{args.dataset}/test_V.npy")
    E = [e[:, :args.K] for e in E]

    ckpts_dir = f"./ckpts/{args.dataset}/gf/"

    model = GF(embedding_dim = args.K, num_flows = 2, device = "cpu")
    model.load_state_dict(torch.load(f"{ckpts_dir}/weights.torch"))

    dataset = GraphDataset(E, A, device = "cpu")
    dataloader = DataLoader(dataset, batch_size = 128, shuffle = True, 
                            collate_fn = custom_collate_fn)

    # compute generated graphs and compare quality
    gen_graphs = generate_for_test_set(model, dataloader, args.batch_size)
    np.save(f"{ckpts_dir}/gen_graphs.npy", gen_graphs)
    stats = {}

    for i in range(len(A)):
        A[i][np.arange(len(A[i])), np.arange(len(A[i]))] = 0
    gen = [nx.from_numpy_array(g) for g in gen_graphs]
    ref = [nx.from_numpy_array(a) for a in A]
    print("== Orbit")
    stats["orbit"] = orbit_stats(ref, gen)
    print(stats["orbit"])
    print("== Degree")
    stats["degree"] = degree_stats(ref, gen)
    print(stats["degree"])
    print("== Cluster")
    stats["cluster"] = cluster_stats(ref, gen)
    print(stats["cluster"])

    # == calculate log-likelihood statistics
    node_bpd, edge_bpd = compute_test_bpd(model, dataloader)
    print("== Embeddings BPD [bits]")
    print(f"{np.mean(node_bpd)} \pm {np.std(node_bpd) / len(node_bpd) ** 0.5}")
    print("== Edges BPD [bits]")
    print(f"{np.mean(edge_bpd)} \pm {np.std(edge_bpd) / len(edge_bpd) ** 0.5}")
    stats["node_bpd_mean"] = float(np.mean(node_bpd))
    stats["node_bpd_stderr"] = float(np.std(node_bpd) / len(node_bpd) ** 0.5)
    stats["edge_bpd_mean"] = float(np.mean(edge_bpd))
    stats["edge_bpd_stderr"] = float(np.std(edge_bpd) / len(edge_bpd) ** 0.5)

    with open(f"{ckpts_dir}/stats.json", "w") as statsfile:
        statsfile.write(json.dumps(stats))

    # == create the figures using entire dataset
    plot_prior_histograms(model, dataloader)
    plt.savefig(f"{ckpts_dir}/prior.png")

    # == create the figures using subsamples
    idx = np.random.choice(len(E) - 8)
    dataset = GraphDataset(E[idx : idx + 8], A[idx : idx + 8], device = "cpu")
    dataloader = DataLoader(dataset, batch_size = 128, shuffle = True,
                            collate_fn = custom_collate_fn)

    plot_sample_graphs(model, dataloader)
    plt.savefig(f"{ckpts_dir}/samples.png")

    plot_reconstructions(model, dataloader)
    plt.savefig(f"{ckpts_dir}/reconstructions.png")

    # == create the figures using generated data
    plot_generations(n_batch = 8, n_nodes = int(np.round(np.mean(V))))
    plt.savefig(f"{ckpts_dir}/generations.png")

    # == create the figures using subsamples of same length
    mean_n_nodes = int(np.round(np.mean(V)))
    E_sub = list(filter(lambda e: len(e) == mean_n_nodes, E))[:4]
    A_sub = list(filter(lambda a: len(a) == mean_n_nodes, A))[:4]
    if len(A_sub) == 4:
        dataset = GraphDataset(E_sub, A_sub, device = "cpu")
        dataloader = DataLoader(dataset, batch_size = 128, shuffle = True,
                                collate_fn = custom_collate_fn)
        plot_interpolations(model, dataloader)
        plt.savefig(f"{ckpts_dir}/interpolations.png")

