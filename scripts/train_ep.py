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
    argparser.add_argument("--lr", default=0.01, type=float)
    argparser.add_argument("--iterations", default=4000, type=int)
    argparser.add_argument("--batch-size", default=2000, type=int)
    argparser.add_argument("--device", default="cuda:0")
    argparser.add_argument("--noise", default=0.01, type=float)
    argparser.add_argument("--load", action="store_true")
    args = argparser.parse_args()

    logging.basicConfig(level = logging.DEBUG)
    logger = logging.getLogger(__name__)
    
    E = np.load(f"datasets/{args.dataset}/train_E.npy", allow_pickle = True)
    A = np.load(f"datasets/{args.dataset}/train_A.npy", allow_pickle = True)
    V = np.load(f"datasets/{args.dataset}/train_V.npy")

    E = [e[:, :args.K] for e in E]
    E = [e + args.noise * np.random.randn(*e.shape) for e in E]
    mu = np.mean(np.vstack([e[:,:args.K] for e in E]), axis = 0)
    sigma = np.std(np.vstack([e[:,:args.K] for e in E]), axis = 0)
    E = [(e - mu) / sigma for e in E]
    E, A = zip(*sorted(zip(E, A), key = lambda t: len(t[0])))

    #_, X, Y = convert_embeddings_pairwise(E, A)
    edge_data = EdgeDataset(E, A, device = "cpu")
    sampler = CustomBatchSampler(edge_data, batch_size = args.batch_size)
    dataloader = DataLoader(edge_data, batch_sampler = sampler)
    iterator = iter(dataloader)

    edge_predictor = EdgePredictor(args.K, device = args.device)
    optimizer = optim.Adam(edge_predictor.parameters(), lr = args.lr)
    epoch_no = 1

    if args.load:
        edge_predictor.load_state_dict(torch.load("./ckpts/ep/weights.torch")) 
        print("Successfully loaded model.")


    for i in range(args.iterations):
        optimizer.zero_grad()
        try:
            X_batch, Y_batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            X_batch, Y_batch = next(iterator)
            epoch_no += 1
        loss = edge_predictor.loss(X_batch, Y_batch).mean()
        loss.backward()
        optimizer.step()
        if i % 1 == 0:
            logger.info(f"Iter: {i}\t" + \
                        f"Loss: {loss.mean().data:.2f}\t" + \
                        f"Batch size: {len(X_batch)}\t" + \
                        f"Epoch: {epoch_no}")

    torch.save(edge_predictor.state_dict(), f"./ckpts/ep/weights.torch")

