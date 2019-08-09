import numpy as np
from matplotlib import pyplot as plt
from gf.utils import compute_fgsd_embeddings
from argparse import ArgumentParser
from datasets.community.gen_community import gen_graphs


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--dataset", default="community")
    argparser.add_argument("--K", default=10, type=int)
    args = argparser.parse_args()
    
    E = np.load(f"datasets/{args.dataset}/train_E.npy", allow_pickle = True)
    mu = np.mean(np.vstack([e[:,:args.K] for e in E]), axis = 0)
    sigma = np.std(np.vstack([e[:,:args.K] for e in E]), axis = 0)
    idxs = np.random.choice(len(E), 3, replace = False)
    E = E[idxs]

    plt.figure(figsize=(10, 8))
    for i, e in enumerate(E):
        plt.subplot(2, 3, i + 1)
        plt.imshow(e[:,:args.K], vmin = -1., vmax = 1.)
        plt.colorbar()
        plt.subplot(2, 3, i + 4)
        plt.imshow((e[:,:args.K] - mu) / sigma)
        plt.colorbar()
    plt.show()
    plt.savefig("./ckpts/img/fgsd.png")

