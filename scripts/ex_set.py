import numpy as np
import itertools
import logging
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn.functional as F
from argparse import ArgumentParser
from torch.distributions import Normal

from gf.models.gf import *
from sklearn.datasets import make_moons


def gen_data(n=16, d=50):
    X = []
    for _ in range(n):
        P = np.eye(d)
        X += [P @ make_moons(d)[0]]
    X = np.array(X)
    return X + 0.05 * np.random.randn(*X.shape)

def plot_data(x, **kwargs):
    plt.scatter(x[:,:,0], x[:,:,1], marker="x", **kwargs)
    plt.xlim((-3, 3))
    plt.ylim((-3, 3))


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--n", default=16, type=int)
    argparser.add_argument("--d", default=50, type=int)
    argparser.add_argument("--flows", default=2, type=int)
    argparser.add_argument("--iterations", default=500, type=int)
    argparser.add_argument("--use-mixture", action="store_true")
    argparser.add_argument("--convolve", action="store_true")
    argparser.add_argument("--actnorm", action="store_true")
    args = argparser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    flows = [GFLayerNSF(embedding_dim=2, device="cpu") for _ in range(args.flows)]
    prior = MultivariateNormal(torch.zeros(2), torch.eye(2))
    model = nn.ModuleList(flows)

    optimizer = optim.Adam(model.parameters(), lr=0.005)
    x = torch.Tensor(gen_data(args.n))
    v = args.d * torch.ones(args.n)
    prior = Normal(torch.zeros(args.d, 2), torch.ones(args.d, 2))

    for i in range(args.iterations):
        optimizer.zero_grad()
        log_det = torch.zeros_like(v)
        z = x
        for flow in flows:
            z, ld = flow.forward(z, v)
            log_det += ld
        prior_logprob = prior.log_prob(z).sum(dim=(1, 2))
        loss = -torch.mean((prior_logprob + log_det) / (args.d * 2))
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            logger.info(f"Iter: {i}\t" +
                        f"Prior: {prior_logprob.mean().data:.2f}\t" +
                        f"LogDet: {log_det.mean().data:.2f}")

    samples = prior.sample([16])
    for flow in flows[::-1]:
        samples, _ = flow.backward(samples, v)
    samples = samples.data.numpy()

    plt.figure(figsize=(8, 3))
    plt.subplot(1, 3, 1)
    plot_data(x[0], color="black", alpha=0.5)
    plt.title("Training data")
    plt.subplot(1, 3, 2)
    plot_data(z.data, color="darkblue", alpha=0.5)
    plt.title("Latent space")
    plt.subplot(1, 3, 3)
    plot_data(samples, color="black", alpha=0.5)
    plt.title("Generated samples")
    plt.savefig("./examples/ex_2d.png")
    plt.show()

    for f in flows:
        x = f(x, v)[0].data
        plot_data(x, color="black", alpha=0.5)
        plt.show()

