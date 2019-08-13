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
from gf.models.gf import GF
from gf.utils import *
from gf.datasets import *
from tqdm import tqdm
mpl.use("agg")


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--dataset", default="community")
    argparser.add_argument("--K", default=4, type=int)
    argparser.add_argument("--lr", default=1e-4, type=float)
    argparser.add_argument("--iterations", default=1000, type=int)
    argparser.add_argument("--device", default="cuda:0")
    argparser.add_argument("--batch-size", default=2000, type=int)
    argparser.add_argument("--noise", default=0.025, type=float)
    argparser.add_argument("--load", action="store_true")
    argparser.add_argument("--ep-only", action="store_true")
    args = argparser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    E = np.load(f"datasets/{args.dataset}/train_E.npy", allow_pickle = True)
    A = np.load(f"datasets/{args.dataset}/train_A.npy", allow_pickle = True)
    V = np.load(f"datasets/{args.dataset}/train_V.npy")
    
    E = [e[:, :args.K] for e in E]
    E = [e + args.noise * np.random.randn(*e.shape) for e in E]

    mu = np.mean(np.vstack(E), axis = 0)
    sigma = np.std(np.vstack(E), axis = 0)
    np.save("./ckpts/gf/mu.npy", mu)
    np.save("./ckpts/gf/sigma.npy", sigma)
    E = [(e - mu) / sigma for e in E]
    E, A = zip(*sorted(zip(E, A), key = lambda t: len(t[0])))

    #dataset = EdgeDataset(E, A, device = args.device)
    #sampler = CustomBatchSampler(dataset, batch_size = args.batch_size)
    #dataloader = DataLoader(dataset, batch_sampler = sampler)

    dataset = GraphDataset(E, A, device = args.device)
    dataloader = DataLoader(dataset, batch_size = 256, shuffle = True,
                            collate_fn = custom_collate_fn)
    iterator = iter(dataloader)

    model = GF(embedding_dim = args.K, num_flows = 2, device = args.device)

    if args.load:
        model.load_state_dict(torch.load("./ckpts/gf/weights.torch"))

    model = model.to(args.device)
    if args.ep_only:
        optimizer = optim.Adam(model.ep.parameters(), lr=args.lr, weight_decay = 1e-5)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay = 1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.5)

    losses = np.zeros(args.iterations)
    epoch_no = 1

    for i in range(args.iterations):
        optimizer.zero_grad()
        try:
            L_batch, A_batch, V_batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            L_batch, A_batch, V_batch = next(iterator)
            epoch_no += 1
        Z, logprob = model.forward(L_batch, A_batch, V_batch)
        loss = -torch.mean(logprob)
        loss.backward()
        losses[i] = loss.cpu().data.numpy()
        optimizer.step()
        #scheduler.step(losses[i])
        if i % 1 == 0:
            logger.info(f"Iter: {i}\t" +
                        f"Logprob: {-loss.data:.2f}\t" + 
                        f"Epoch: {epoch_no}\t" + 
                        f"Batch size: {len(V_batch)}\t" +
                        f"Max nodes: {len(L_batch[0])}")

    model = model.cpu()
    torch.save(model.state_dict(), "./ckpts/gf/weights.torch")
    np.save("./ckpts/gf/loss_curve.npy", losses)
    
    plt.figure(figsize=(8, 5))
    losses = np.load("./ckpts/gf/loss_curve.npy")
    plt.plot(np.arange(len(losses)) + 1, losses, color = "black", alpha = 0.5)
    plt.title("Training loss")
    plt.savefig("./ckpts/gf/loss_curve.png")

