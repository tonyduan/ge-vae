import numpy as np
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from datasets.community.gen_community import gen_graphs


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--dataset", default="community")
    argparser.add_argument("--K", default=10, type=int)
    argparser.add_argument("--N", default=8, type=int)
    args = argparser.parse_args()

    E = np.load(f"datasets/{args.dataset}/train_E.npy", allow_pickle = True)
    idxs = np.random.choice(len(E), args.N, replace = False)
    E = E[idxs]

    plt.figure(figsize=(10, 8))
    for i, e in enumerate(E):
        plt.subplot(2, args.N, i + 1)
        plt.imshow(e[:, :args.K], vmin = -1., vmax = 1.)
        plt.subplot(2, args.N, i + args.N + 1)
        plt.scatter(e[:, 0], e[:, 1], color = "black")
    plt.show()

