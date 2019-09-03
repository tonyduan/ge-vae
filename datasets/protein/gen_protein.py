import pandas as pd
import networkx as nx
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
from gf.utils import *
from gf.embeddings import *


def load_protein_data(min_num_nodes = 20, max_num_nodes = 1000):
    '''
    load many graphs, e.g. enzymes
    :return: a list of graphs
    '''
    G = nx.Graph()
    adj = np.loadtxt("datasets/protein/DD_A.txt", delimiter=',')
    graph_indicator = np.loadtxt("datasets/protein/DD_graph_indicator.txt",
                                 delimiter=',').astype(int)
    data_tuple = list(map(tuple, adj))
    G.add_edges_from(data_tuple)
    graph_num = graph_indicator.max()
    node_list = np.arange(graph_indicator.shape[0])+1
    graphs = []
    max_nodes = 0
    for i in tqdm(range(graph_num)):
        nodes = node_list[graph_indicator==i+1]
        G_sub = G.subgraph(nodes)
        if G_sub.number_of_nodes()>=min_num_nodes and \
           G_sub.number_of_nodes()<=max_num_nodes:
            graphs.append(G_sub)
            if G_sub.number_of_nodes() > max_nodes:
                max_nodes = G_sub.number_of_nodes()
    A = [nx.to_numpy_array(g) for g in graphs]
    V = [len(a) for a in A]
    return np.array(V), np.array(A)


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--train-N", default=880, type=int)
    argparser.add_argument("--seed", default=123, type=int)
    args = argparser.parse_args()

    np.random.seed(args.seed)

    V, A = load_protein_data()
    E = np.array([compute_fgsd_embeddings(a) for a in A])
    K = min([e.shape[1] for e in E])
    E = np.array([e[:, :K] for e in E])

    train_idxs = np.random.choice(len(V), args.train_N, replace = False)
    test_idxs = np.setdiff1d(np.arange(len(V)), train_idxs)

    np.save("datasets/protein/train_A.npy", A[train_idxs])
    np.save("datasets/protein/train_E.npy", E[train_idxs])
    np.save("datasets/protein/train_V.npy", V[train_idxs])

    np.save("datasets/protein/test_A.npy", A[test_idxs])
    np.save("datasets/protein/test_E.npy", E[test_idxs])
    np.save("datasets/protein/test_V.npy", V[test_idxs])
