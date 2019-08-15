import logging
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networkx as nx
import matplotlib as mpl
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from torch.utils.data.dataloader import DataLoader
import json
from gf.models.gf import GF
from gf.utils import *
from gf.datasets import *
from tqdm import tqdm
from pathlib import Path
mpl.use("agg")


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--dataset", default="community", type=str)
    argparser.add_argument("--K", default=4, type=int)
    argparser.add_argument("--n-flows", default=2, type=int)
    argparser.add_argument("--lr", default=0.001, type=float)
    argparser.add_argument("--iterations", default=15000, type=int)
    argparser.add_argument("--device", default="cuda:0", type=str)
    argparser.add_argument("--batch-size", default=256, type=int)
    argparser.add_argument("--print-every", default=1, type=str)
    argparser.add_argument("--load", action="store_true")
    args = argparser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    ckpts_dir = f"./ckpts/{args.dataset}/gf/"
    Path(ckpts_dir).mkdir(parents = True, exist_ok = True)

    E = np.load(f"datasets/{args.dataset}/train_E.npy", allow_pickle = True)
    A = np.load(f"datasets/{args.dataset}/train_A.npy", allow_pickle = True)
    V = np.load(f"datasets/{args.dataset}/train_V.npy")
    E = [e[:, :args.K] for e in E]

    dataset = GraphDataset(E, A, device = args.device)
    dataloader = DataLoader(dataset, batch_size = args.batch_size, 
                            shuffle = True, collate_fn = custom_collate_fn)
    iterator = iter(dataloader)

    model = GF(embedding_dim = args.K, num_flows = args.n_flows, 
               device = args.device)

    if args.load:
        model.load_state_dict(torch.load("./ckpts/gf/weights.torch"))

    model = model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay = 1e-5)

    losses = np.zeros(args.iterations)
    epoch_no = 1

    for i in range(args.iterations):

        try:
            E_batch, A_batch, V_batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            E_batch, A_batch, V_batch = next(iterator)
            epoch_no += 1

        optimizer.zero_grad()
        _, node_lp, edge_lp = model.forward(E_batch, A_batch, V_batch)
        loss = -torch.mean(node_lp + edge_lp)
        loss.backward()
        losses[i] = loss.cpu().data.numpy()
        optimizer.step()

        if i % args.print_every == 0:
            logger.info(f"Iter: {i}\t" +
                        f"Node LP: {node_lp.mean().data:.2f}\t" + 
                        f"Edge LP: {edge_lp.mean().data:.2f}\t" + 
                        f"Epoch: {epoch_no}\t" + 
                        f"Batch size: {len(V_batch)}\t" +
                        f"Max nodes: {len(A_batch[0])}")

    model = model.cpu()
    torch.save(model.state_dict(), f"{ckpts_dir}/weights.torch")
    
    with open(f"{ckpts_dir}/args.json", "w") as argsfile:
        argsfile.write(json.dumps(args.__dict__))

    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(len(losses)) + 1, losses, color = "black", alpha = 0.5)
    plt.title("Training loss")
    plt.savefig(f"{ckpts_dir}/loss_curve.png")

