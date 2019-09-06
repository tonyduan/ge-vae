# -*- coding: future_fstrings -*-
"""
This file contains utility functions for working with graphs.
"""
import numpy as np
import networkx as nx
import torch
import itertools
from tqdm import tqdm


def construct_embedding_mask(V):
    """
    Construct a mask for a batch of embeddings given node sizes.

    Parameters
    ----------
    V: (batch_size) actual number of nodes per set (tensor)

    Returns
    -------
    mask: (batch_size) x (n_nodes) binary mask (tensor)
    """
    batch_size = len(V)
    max_n_nodes = torch.max(V).int()
    mask = torch.zeros(batch_size, max_n_nodes, device = str(V.device))
    for i, cnt in enumerate(V):
        mask[i, :cnt.int()] = 1
    return mask


def construct_adjacency_mask(V):
    """
    Construct a mask for a batch of adjacency matrices given node sizes.

    Parameters
    ----------
    V: (batch_size) actual number of nodes per set (tensor)

    Returns
    -------
    mask: (batch_size) x (n_nodes) binary mask (tensor)
    """
    batch_size = len(V)
    max_n_nodes = torch.max(V).int()
    mask = torch.zeros(batch_size, max_n_nodes, max_n_nodes,
                       device = str(V.device))
    for i, cnt in enumerate(V):
        mask[i, :cnt.int(), :cnt.int()] = 1
    return mask


def convert_embeddings_pairwise(E, A=None):
    """
    Convert to a representation with a pairwise relationship between each node.

    First and second row represent embeddings for the pair in question.
    Rest of the rows represent embeddings for all remaining pairs.

    Parameters
    ----------
    E: (batch_size) x (n_nodes) x (n_features) set of embeddings

    A: (batch_size) x (n_nodes) x (n_nodes) adjacency matrix entries {0, 1}
        if A is None, that means we don't have labels

    Returns
    -------
    idxs: (batch_size x n_nodes choose 2) tuples (i, j) of node pair indices

    X: (batch_size x n_nodes choose 2) x (n_nodes) x (n_features) embeddings

    Y: (batch_size x n_nodes choose 2) labels either {0, 1}
        only returned if A is not None
    """
    X, Y, idxs = [], [], []
    for b, E_k in tqdm(enumerate(E), total=len(E)):
        for (i, j) in itertools.combinations(np.arange(len(E_k)), 2):
            first = E_k[i][np.newaxis,:]
            second = E_k[j][np.newaxis,:]
            rest_idx = np.r_[np.arange(i), np.arange(i + 1, j),
                             np.arange(j + 1, len(E_k))]
            rest = np.take(E_k, rest_idx, axis = 0)
            idxs += [(i, j)]
            X += [np.r_[first, second, rest]]
            if A is not None:
                Y += [A[b][i, j]]
    if A is None:
        return idxs, np.array(X)
    else:
        return idxs, np.array(X), np.array(Y)


def reconstruct_adjacency_matrix(N, idxs, Y_hat):
    """
    Reconstruct the adjacency matrix from a set of indices and predictions.

    Parameters
    ----------
    N: number of nodes

    idxs: (n_nodes choose 2) tuples (i, j) of node pair indices

    Y_hat: (n_nodes choose 2) predictions for each pair (i, j)
    """
    A = np.zeros((N, N))
    for (i, j), y in zip(idxs, Y_hat):
        A[i,j] = y
        A[j,i] = y
    return A