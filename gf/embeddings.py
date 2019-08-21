import itertools
import numpy as np
import cvxpy as cp
import subprocess
from gf.utils import *


def compute_unnormalized_laplacian_eigenmaps(A, eps=1e-2):
    """
    Compute unnormalized Laplacian eigenmap embedding of a graph.

    Note that since eigenvalues can repeat this approach is not recommended
    to construct permutation-invariant embeddings.

    L = D - A = Z Λ Zᵀ

    Returns the vectors Z.

    Parameters
    ----------
    A: (n_nodes x n_nodes) adjacency matrix

    Returns
    -------
    E: (n_nodes x n_nodes) Laplacian embeddings
    """
    D = np.diag(np.sum(A, axis=1))
    L = D - A
    W, E = np.linalg.eigh(L)
    E = E @ np.diag(np.sign(np.sum(E, axis = 0))) 
    return np.real(E)

def compute_normalized_laplacian_eigenmaps(A, eps=1e-2):
    """
    Compute normalized Laplacian eigenmap embedding of a graph.

    L = D - A = Z Λ Zᵀ

    Returns the vectors D^(-½) Z, which scales each row by inverse sqrt degree. 

    Parameters
    ----------
    A: (n_nodes x n_nodes) adjacency matrix

    Returns
    -------
    E: (n_nodes x n_nodes) Laplacian embeddings
    """
    D = np.sum(A, axis=1)
    L = np.diag(D) - A
    L = np.diag(D ** -0.5) @ L @ np.diag(D ** -0.5)
    W, E = np.linalg.eigh(L)
    E = E @ np.diag(np.sign(np.sum(E, axis = 0))) 
    return np.real(E)

def compute_locally_linear_embedding(A):
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
    return (np.diag(np.sign(np.sum(U, axis = 0))) @ V).T
   
def compute_structure_preserving_embedding(A, C = 100.0, verbose = False):
    """
    Compute the structure-preserving embedding of a graph using CVXPY.

    This does not actually scale beyond 10 x 10 or so.
    
    Parameters
    ----------
    A: (n_nodes x n_nodes) adjacency matrix

    Returns
    -------
    E: (n_nodes x n_nodes) LLE embeddings
    """
    n_nodes, _ = A.shape

    K = cp.Variable((n_nodes, n_nodes), PSD = True)
    xi = cp.Variable(1)

    D = cp.bmat([[K[i, i] + K[j, j] - 2 * K[i, j] \
                  for j in range(n_nodes)] \
                  for i in range(n_nodes)])
    M = cp.max(D, axis = 1)
    
    objective = cp.Maximize(cp.trace(K @ A) - C * xi)
    constraints = [cp.trace(K) <= 1, xi >= 0, cp.sum(K) == 0]
    for i, j in itertools.combinations(range(n_nodes), 2):
        constraints += [D[i, j] >= (1 - A[i, j]) * M[i] - xi]

    problem = cp.Problem(objective, constraints)
    result = problem.solve(verbose = verbose, solver = "SCS")
    
    L, Z = np.linalg.eigh(K.value)
    return Z

def compute_node2vec_embedding(A, dim = 64, walk_length = 80):
    """
    Wrapper around Stanford's SNAP node2vec C++ implementation.

    Parameters
    ----------
    A: (n_nodes x n_nodes) adjacency matrix

    Returns
    -------
    E: (n_nodes x n_nodes) node2vec embeddings
    """
    write_adjacency_matrix_to_edge_list(A, "/tmp/node2vec_edge_list.graph")

    args = ["./lib/node2vec", 
            "-i:/tmp/node2vec_edge_list.graph", 
            "-o:/tmp/node2vec_embeddings.emb",
            "-v:no", 
            f"-d:{dim}",
            f"-l:{walk_length}"]
    subprocess.run(args, stdout = subprocess.DEVNULL)

    return load_embedding_from_file("/tmp/node2vec_embeddings.emb")
        
