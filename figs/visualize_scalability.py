import numpy as np
import networkx as nx
from argparse import ArgumentParser
from matplotlib import pyplot as plt


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--method", default="laplacian")
    argparser.add_argument("--dataset", default="grid")
    argparser.add_argument("--cnt", default=3, type=int)
    args = argparser.parse_args()

    plt.figure(figsize = (12, 4))
    for i in range(args.cnt):

        if args.dataset == "grid":
            A = nx.to_numpy_array(nx.grid_2d_graph(i + 2, i + 2))
        if args.dataset == "ladder":
            A = nx.to_numpy_array(nx.ladder_graph(i + 4))
        if args.method == "svd":
            U, S, V = np.linalg.svd(A)
        if args.method == "laplacian":
            S, V = np.linalg.eigh(np.diag(np.sum(A, axis = 0)) - A)
            V = V.T[1:, :]
        if args.method == "lle":
            U, S, V = np.linalg.svd(np.eye(len(A)) - A)

        plt.subplot(3, args.cnt, i + 1)
        plt.imshow(V.T)
        plt.axis("off")
        plt.subplot(3, args.cnt, i + args.cnt + 1)
        nx.draw_spring(nx.from_numpy_array(A), node_size = 20,
                       node_color = "black")
        plt.axis("off")
        plt.subplot(3, args.cnt, i + 2 * args.cnt + 1)
        plt.scatter(V.T[:, 1], V.T[:, 0], color = "black")
        for i in range(len(A)):
            for j in range(i, len(A)):
                if A[i,j]:
                    plt.plot([V.T[i, 1], V.T[j, 1]],
                             [V.T[i, 0], V.T[j, 0]],
                             color = "black", alpha = 0.2)
        plt.axis("off")

    plt.tight_layout()
    plt.show()
