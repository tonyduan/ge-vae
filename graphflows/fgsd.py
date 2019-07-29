import numpy as np
import scipy as sp
import scipy.spatial
import networkx as nx
from matplotlib import pyplot as plt


def compute_fgsd_embeddings(A, eps=1e-2):
    D = np.diag(np.sum(A, axis=1))
    L = D - A
    W, V = np.linalg.eigh(L)
    if np.mean(V[:,0]) < 0: # todo: make this less hacky
        V[:,0] = -V[:,0]
    W[W < eps] = 0.1
    W[W > eps] = 1 / W[W > eps]
    E = V @ np.diag(np.sqrt(W))
    return np.real(E)

def compute_fgsd_dists(A, eps=1e-2):
    D = np.diag(np.sum(A, axis=1))
    L = D - A
    W, V = np.linalg.eig(L)
    W_orig = W.copy()
    W[W < eps] = 0
    W[W > eps] = 1 / W[W > eps]
    J = np.ones_like(A) @ np.ones_like(A).T
    L = V @ np.diag(W) @ V.T
    return np.diag(L) @ J + J @ np.diag(L) - 2 * L

def invert_fgsd_embeddings(E, eps=1e-2):
    dists = sp.spatial.distance.cdist(E, E, metric="sqeuclidean")
    breakpoint()
