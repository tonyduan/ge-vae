import numpy as np
import torch
import json
import networkx as nx
import matplotlib as mpl
import pandas as pd
import scipy as sp
import scipy.stats
from argparse import ArgumentParser
from src.models.gevae import GEVAE
from src.models.ep import EdgePredictor
from src.datasets import *
from src.utils import *
from src.eval.stats import *
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from matplotlib import pyplot as plt
mpl.use("agg")


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
    plt.figure(figsize = (embedding_dim, 6))
    for i in range(embedding_dim):
        plt.subplot(3, embedding_dim // 2, i + 1)
        density = sp.stats.gaussian_kde(z[:, i], "silverman")
        plt.plot(x_axis, density(x_axis))
        plt.fill_between(x_axis, 0, density(x_axis), alpha=0.2)
        plt.title(f"Dimension {i + 1}")
    plt.subplot(3, embedding_dim // 2, embedding_dim + 1)
    density = sp.stats.gaussian_kde(z.flatten(), "silverman")
    plt.plot(x_axis, density(x_axis))
    plt.fill_between(x_axis, 0, density(x_axis), alpha=0.2)
    plt.title("Aggregate")
    plt.subplot(3, embedding_dim // 2, embedding_dim + 2)
    density = sp.stats.norm.pdf(x_axis)
    plt.plot(x_axis, density)
    plt.fill_between(x_axis, 0, density, alpha=0.2)
    plt.title("Ideal")
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
        a_hat = model.predict_a_from_e(e, n_nodes).data.numpy()[0]
        z = np.random.rand(*a_hat.shape)
        a_sample = (np.tril(z) + np.tril(z, -1).T < a_hat).astype(int)
        cc = get_largest_cc(a_sample)
        nx.draw(nx.from_numpy_array(cc), node_color = "black",
                node_size = 20, alpha = 0.5)
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
        nx.draw_kamada_kawai(nx.from_numpy_array(cc), node_color = "black", node_size = 20,
                       alpha = 0.3)
    plt.tight_layout()


def plot_gen_embeddings(model, dataloader, n_batch, n_nodes):
    z_sampled, v_sampled = model.sample_prior(n_batch, n_nodes)
    e_sampled = model.backward(z_sampled, v_sampled)
    v_sampled = v_sampled.data.numpy().astype(int)
    plt.figure(figsize = (n_batch * 3, 8))
    for i in range(n_batch):
        e_current = e_sampled[i].data.numpy()[:v_sampled[i], :]
        plt.subplot(4, n_batch, i + 1)
        plt.imshow(e_current)
        plt.subplot(4, n_batch, i + n_batch + 1)
        plt.scatter(e_current[:, 0],  e_current[:, 1], color = "black")
    x_batch, _, v_batch = next(iter(dataloader))
    v_batch = v_batch.data.numpy().astype(int)
    for i in range(n_batch):
        x_current = x_batch[i].data.numpy()[:v_batch[i], :]
        plt.subplot(4, n_batch, i + 2 * n_batch + 1)
        plt.imshow(x_current)
        plt.subplot(4, n_batch, i + 3 * n_batch + 1)
        plt.scatter(x_current[:,  0], x_current[:, 1], color = "black")
    plt.tight_layout()

def plot_reconstructions(model, dataloader):
    x_batch, _, v_batch = next(iter(dataloader))
    a_hat = model.predict_a_from_e(x_batch, v_batch).data.numpy()
    n_graphs = len(x_batch)
    v_batch = v_batch.data.numpy().astype(int)
    plt.figure(figsize=(n_graphs, 4))
    for i in range(n_graphs):
        z = np.random.rand(*a_hat[i].shape)
        a_sample = (np.tril(z) + np.tril(z, -1).T < a_hat[i]).astype(int)
        a_sample = a_sample[:v_batch[i], :v_batch[i]]
        cc = get_largest_cc(a_sample)
        plt.subplot(2, n_graphs // 2, i + 1)
        nx.draw(nx.from_numpy_array(cc), node_color = "black", node_size = 20)
    plt.tight_layout()

def plot_loss_curve(loss_curve):
    plt.figure(figsize=(8, 5))
    x_axis = np.arange(len(loss_curve["edge_lp"])) + 1
    plt.plot(x_axis, -loss_curve["node_lp"], color = "black",
             label = "Node NLL [bits per dim]")
    plt.plot(x_axis, -loss_curve["edge_lp"], color = "grey",
             label = "Edge NLL [bits per dim]")
    plt.ylim(bottom = -3.0)
    plt.legend()
    plt.title("Training loss")

def compute_test_bpd(model, dataloader, n_monte_carlo = 128):
    log_liks = [[]] * n_monte_carlo
    for E_batch, A_batch, V_batch in tqdm(iter(dataloader)):
        cnts = V_batch.data.numpy() * (V_batch.data.numpy() - 1)
        for i in range(n_monte_carlo):
            _, node_lp, edge_lp = model.forward(E_batch, A_batch, V_batch)
            log_liks[i] += [(node_lp.data.numpy()+edge_lp.data.numpy()) / cnts]
    log_liks = [np.concatenate(lst) for lst in log_liks]
    log_liks = -np.log(np.mean(np.exp(np.array(log_liks)), axis = 0))
    return log_liks

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
    argparser.add_argument("--batch-size", default=128, type=int)
    argparser.add_argument("--split", default="test")
    argparser.add_argument("--n-monte-carlo", default = 128, type = int)
    argparser.add_argument("--no-calc-stats", action="store_true")
    args = argparser.parse_args()

    ckpts_dir = f"./ckpts/{args.dataset}"
    args_json = open(f"{ckpts_dir}/args.json", "r")
    args_json = json.loads(args_json.read())

    E = np.load(f"datasets/{args.dataset}/{args.split}_E.npy",
                allow_pickle = True)
    A = np.load(f"datasets/{args.dataset}/{args.split}_A.npy",
                allow_pickle = True)
    V = np.load(f"datasets/{args.dataset}/{args.split}_V.npy")
    E = [e[:, :args_json["K"]] for e in E]

    model = GEVAE(embedding_dim = args_json["K"],
                  num_flows = args_json["n_flows"],
                  noise_lvl = args_json["noise_lvl"],
                  n_knots = args_json["n_knots"],
                  device = "cpu")
    model.load_state_dict(torch.load(f"{ckpts_dir}/weights.torch"))
    model = model.eval()

    dataset = GraphDataset(E, A, device = "cpu")
    dataloader = DataLoader(dataset, batch_size = 128, shuffle = True,
                            collate_fn = custom_collate_fn)

    if not args.no_calc_stats:

        # plot training statistics
        plot_loss_curve(pd.read_csv(f"{ckpts_dir}/loss_curve.csv"))
        plt.savefig(f"{ckpts_dir}/loss_curve.png")

        # compute generated graphs and compare quality
        gen_graphs = generate_for_test_set(model, dataloader, args.batch_size)
        np.save(f"{ckpts_dir}/gen_graphs.npy", gen_graphs)
        stats = {}

        for i in range(len(A)):
            A[i][np.arange(len(A[i])), np.arange(len(A[i]))] = 0
        gen = [nx.from_numpy_array(g) for g in gen_graphs]
        ref = [nx.from_numpy_array(a) for a in A]
        print("== Degree")
        stats["degree"] = degree_stats(ref, gen)
        print(stats["degree"])
        print("== Cluster")
        stats["cluster"] = cluster_stats(ref, gen)
        print(stats["cluster"])
        print("== Orbit")
        stats["orbit"] = orbit_stats(ref, gen)
        print(stats["orbit"])

       # == calculate log-likelihood statistics
        bpds = compute_test_bpd(model, dataloader, args.n_monte_carlo)
        print("== Embeddings BPD [bits]")
        print(f"{np.mean(bpds)} \pm {np.std(bpds) / len(bpds) ** 0.5}")
        stats["bpd_mean"] = float(np.mean(bpds))
        stats["bpd_stderr"] = float(np.std(bpds) / len(bpds) ** 0.5)

        with open(f"{ckpts_dir}/stats_{args.split}.json", "w") as statsfile:
            statsfile.write(json.dumps(stats))

   # == create the figures using entire dataset
    plot_prior_histograms(model, dataloader)
    plt.savefig(f"{ckpts_dir}/prior_{args.split}.png")

    # == create the figures using subsamples
    idx = np.random.choice(len(E) - 8)
    dataset = GraphDataset(E[idx : idx + 8], A[idx : idx + 8], device = "cpu")
    dataloader = DataLoader(dataset, batch_size = 128, shuffle = True,
                            collate_fn = custom_collate_fn)

    plot_sample_graphs(model, dataloader)
    plt.savefig(f"{ckpts_dir}/samples.png")

    plot_reconstructions(model, dataloader)
    plt.savefig(f"{ckpts_dir}/reconstructions.png")

    plot_gen_embeddings(model, dataloader, n_batch = 4,
                        n_nodes = int(np.round(np.mean(V))))
    plt.savefig(f"{ckpts_dir}/embeddings.png")

    # == create the figures using generated data
    #plot_generations(n_batch = 8, n_nodes = int(np.median(V)))
    plot_generations(n_batch = 8, n_nodes = 100)
    plt.savefig(f"{ckpts_dir}/generations.png")

    # == create the figures using subsamples of same length
    mean_n_nodes = int(np.round(np.median(V)))
    E_sub = list(filter(lambda e: len(e) == mean_n_nodes, E))[:4]
    A_sub = list(filter(lambda a: len(a) == mean_n_nodes, A))[:4]
    if len(A_sub) == 4:
        dataset = GraphDataset(E_sub, A_sub, device = "cpu")
        dataloader = DataLoader(dataset, batch_size = 128, shuffle = True,
                                collate_fn = custom_collate_fn)
        plot_reconstructions(model, dataloader)
        plt.savefig(f"{ckpts_dir}/reconstructions.png")
        plot_interpolations(model, dataloader)
        plt.savefig(f"{ckpts_dir}/interpolations.png")

