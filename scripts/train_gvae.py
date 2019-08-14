import logging
import itertools
import torch
import torch.optim as optim
import numpy as np
import networkx as nx
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from gf.models.gvae import GVAE


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--N", default=200, type=int)
    argparser.add_argument("--iterations", default=200, type=int)
    argparser.add_argument("--train", action="store_true")
    args = argparser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    sizes = np.random.choice(np.arange(18, 19), size=args.N)
    X, A = gen_graphs(sizes)
    plot_graphs(X, A[:4])
    A = torch.tensor(A, dtype=torch.float)

    model = GVAE(n_nodes = 18, latent_dim = 4)

    optimizer = optim.Adam(model.parameters(), lr=0.005)

    for i in range(args.iterations):
        optimizer.zero_grad()
        loss = model.loss(A)
        loss.mean().backward()
        optimizer.step()
        if i % 1 == 0:
            logger.info(f"Iter: {i}\t" +
                        f"Loss: {loss.mean().data:.2f}\t")

    Z_recon = model.compute_z_given_a(A).sample()
    A_recon = model.compute_a_given_z(Z_recon).probs

    # histogram of latent samples
    plt.figure(figsize = (8, 3))
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.hist(Z_recon[:, :, i].data.numpy().flatten())
    plt.show()

    # compare reconstructed to observed
    plt.figure(figsize = (8, 9))
    for i in range(4):
        plt.subplot(3, 4, i + 1)
        plt.imshow(Z_recon[i].data.numpy())
        plt.subplot(3, 4, i + 5)
        plt.imshow(A_recon[i].data.numpy())
        plt.subplot(3, 4, i + 9)
        plt.imshow(A[i].data.numpy())
    plt.tight_layout()
    plt.show()

    # check permutation invariance
    Z0 = model.compute_z_given_a(A[:1])
    logprob = torch.mean(model.compute_a_given_z(Z0.loc).log_prob(A[0]))
    print(logprob)

    P = np.eye(18)
    np.random.shuffle(P)

    A0 = A[0].numpy()
    A1 = P @ A[0].numpy() @ P.T

    Z1 = model.compute_z_given_a(torch.tensor([A1], dtype=torch.float))
    Z0 = model.compute_z_given_a(torch.tensor([A0], dtype=torch.float))
    logprob = torch.mean(model.compute_a_given_z(Z1.loc).log_prob(
        torch.tensor([A1], dtype=torch.float)))
    print(logprob)

    plt.figure(figsize=(8, 4))
    plt.subplot(2, 1, 1)
    plt.imshow(Z0.loc[0].data.numpy().T)
    plt.subplot(2, 1, 2)
    plt.imshow(Z1.loc[0].data.numpy().T)
    plt.show()

    # try generating samples
    sampled_z = model.sample_z(4)
    A = model.compute_a_given_z(sampled_z)
    probs = A.probs.data.numpy()
    plt.imshow(probs[0])
    plt.title("Sampled A [probabilities]")
    plt.colorbar()
    plt.show()
    plt.imshow((np.random.rand(*probs.shape) < probs).astype(int)[0])
    plt.title("Sampled A [sample]")
    plt.show()
    plot_graphs(X, (np.random.rand(*probs.shape) < probs).astype(int))

