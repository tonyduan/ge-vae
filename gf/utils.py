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


def compute_fgsd_embeddings_old(A, eps=1e-2):
    """
    Compute family of graph spectral distance (FGSD) embedding of a graph.

    Parameters
    ----------
    A: (n_nodes x n_nodes) adjacency matrix

    Returns
    -------
    E: (n_nodes x n_nodes) Laplacian embeddings
    """
    D = np.sum(A, axis=1)
    L = np.diag(D) - A
    W, V = np.linalg.eigh(L)
    if np.mean(V[:,0]) < 0: # due to numerical imprecision
        V[:,0] = -V[:,0]
    if np.std(V[:,0] > 0):
        breakpoint()
    W[W < eps] = 1 / len(A) 
    W[W > eps] = 1 / W[W > eps]
    E = V @ np.diag(np.sqrt(W))
    E = np.c_[D, np.ones_like(D) * len(A), E]
    return np.real(E)


def compute_fgsd_embeddings(A):
    """
    Compute locally linear embedding of a graph.

    (I - A)ᵀ(I -A) = Z Λ Zᵀ

    Returns the vectors Z, which solves the LLE problem.

    Parameters
    ----------
    A: (n_nodes x n_nodes) adjacency matrix

    Returns
    -------
    E: (n_nodes x n_nodes) LLE embeddings
    """
    U, S, V = np.linalg.svd(np.eye(len(A)) - A)
    V = np.diag(np.sign(np.sum(U, axis = 0))) @ V
    D = np.sum(A, axis=1)
    E = np.c_[D, np.ones_like(D) * len(A), V.T]
    return E


def compute_fgsd_dists(A, eps=1e-2):
    """
    Compute family of graph spectral distances (FGSD) distances for a graph.

    Parameters
    ----------
    A: (n_nodes x n_nodes) adjacency matrix

    Returns
    -------
    dists: (n_nodes x n_nodes) symmetric pairwise spectral distances
    """
    D = np.diag(np.sum(A, axis=1))
    L = D - A
    W, V = np.linalg.eig(L)
    W[W < eps] = 0
    W[W > eps] = 1 / W[W > eps]
    J = np.ones_like(A) @ np.ones_like(A).T
    L = V @ np.diag(W) @ V.T
    dists = np.diag(L) @ J + J @ np.diag(L) - 2 * L
    return dists

def write_adjacency_matrix_to_edge_list(A, file_name):
    graph = nx.from_numpy_array(A)
    with open(file_name, "w") as f:
        for i, j, w in graph.edges(data = "weight", default = 1):
            f.write(f"{i} {j} {w:.1f}\n")

def load_embedding_from_file(file_name):
    return np.loadtxt(file_name, skiprows = 1)
