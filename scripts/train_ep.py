import logging
import torch
import torch.optim as optim
import numpy as np
from argparse import ArgumentParser
from torch.utils.data.dataloader import DataLoader
from gf.utils import *
from gf.datasets import EdgeDataset, CustomBatchSampler
from gf.models.ep import EdgePredictor


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--K", default=4, type=int)
    argparser.add_argument("--dataset", default="community")
    argparser.add_argument("--lr", default=0.005, type=float)
    argparser.add_argument("--iterations", default=30000, type=int)
    argparser.add_argument("--device", default="cuda:0")
    argparser.add_argument("--noise", default=0.05, type=float)
    args = argparser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    
    E = np.load(f"datasets/{args.dataset}/train_E.npy", allow_pickle = True)
    A = np.load(f"datasets/{args.dataset}/train_A.npy", allow_pickle = True)
    V = np.load(f"datasets/{args.dataset}/train_V.npy")

    E = [e[:, :args.K] for e in E]
    E = [e + args.noise * np.random.randn(*e.shape) for e in E]

    _, X, Y = convert_embeddings_pairwise(E, A)
    edge_data = EdgeDataset(X, Y)
    sampler = CustomBatchSampler(edge_data, batch_size = 500)
    dataloader = DataLoader(edge_data, batch_sampler = sampler)
    iterator = iter(dataloader)

    edge_predictor = EdgePredictor(args.K)
    optimizer = optim.Adam(edge_predictor.parameters(), lr = args.lr)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.5)

    for i in range(args.iterations):
        optimizer.zero_grad()
        try:
            X_batch, Y_batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            X_batch, Y_batch = next(iterator)
        loss = edge_predictor.loss(X_batch, Y_batch).mean()
        loss.backward()
        optimizer.step()
        #scheduler.step(loss.data.numpy())
        if i % 1 == 0:
            logger.info(f"Iter: {i}\t" + \
                        f"Loss: {loss.mean().data:.2f}\t")

    torch.save(edge_predictor.state_dict(), f"./ckpts/ep/weights.torch")

