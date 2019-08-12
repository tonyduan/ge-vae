import torch
import numpy as np
import networkx as nx
import matplotlib as mpl
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from gf.models.ep import EdgePredictor
from gf.utils import *


if __name__ == "__main__":
    

    argparser = ArgumentParser()
    argparser.add_argument("--dataset", default="community")
    argparser.add_argument("--K", default=4, type=int)
    args = argparser.parse_args()

    E = np.load(f"datasets/{args.dataset}/test_E.npy", allow_pickle = True)
    A = np.load(f"datasets/{args.dataset}/test_A.npy", allow_pickle = True)
    V = np.load(f"datasets/{args.dataset}/test_V.npy")

    edge_predictor = EdgePredictor(args.K, device = "cpu")
    edge_predictor.load_state_dict(torch.load("./ckpts/ep/weights.torch"))

    E = [e[:, :args.K] for e in E]
    E = [e + 0.01 * np.random.randn(*e.shape) for e in E]
    mu = np.mean(np.vstack([e[:,:args.K] for e in E]), axis = 0)
    sigma = np.std(np.vstack([e[:,:args.K] for e in E]), axis = 0)
    E = [(e - mu) / sigma for e in E]

    plt.figure(figsize=(6, 8))
    for i in range(8):
        #idxs, X, _ = convert_embeddings_pairwise([E[i]], [A[i]])
        #X = torch.tensor(X, dtype=torch.float)
        #Y_hat = torch.sigmoid(edge_predictor.forward(X)).data.numpy()
        #A_hat = reconstruct_adjacency_matrix(X.shape[1], idxs, Y_hat)
        e = torch.tensor([E[i]], dtype = torch.float)
        A_hat = torch.sigmoid(edge_predictor.forward(e))[0].data.numpy()
        A_sample = (np.random.rand(*A_hat.shape) < A_hat).astype(int)
        plt.subplot(4, 2, i + 1)
        nx.draw(nx.from_numpy_array(A_sample), node_color = "black",
                node_size = 20)
    plt.show()
    plt.savefig("./ckpts/img/ep.png")
    
