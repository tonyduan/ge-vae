import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from gf.utils import compute_fgsd_embeddings
from argparse import ArgumentParser
from datasets.community.gen_community import gen_graphs


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--dataset", default="community")
    argparser.add_argument("--K", default=50, type=int)
    args = argparser.parse_args()
    
    A = np.load(f"datasets/{args.dataset}/train_A.npy", allow_pickle = True)
    idxs = np.random.choice(len(A), 6, replace = False)
    A = A[idxs]

    plt.figure(figsize=(20, 8))
    for i, a in enumerate(A):
        plt.subplot(2, 3, i + 1)
        nx.draw(nx.from_numpy_array(a), node_color = "black", node_size = 20)
    plt.show()
    plt.savefig("./ckpts/img/graphs.png")

