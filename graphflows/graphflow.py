import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.distributions import Normal, Bernoulli, kl, MultivariateNormal
from graphflows.attn import ISAB, PMA


class EdgePredictor(nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()
        self.rest_query = MAB(embedding_dim, embedding_dim, embedding_dim, 1)
        self.pair_query = PMA(embedding_dim, num_heads = 1, num_seeds = 1)
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, X):
        pair = X[:, :2, :]
        rest = X[:, 2:, :]
        return self.fc(self.pair_query(self.rest_query(pair, rest)))

    def loss(self, X, Y):
        return -Bernoulli(logits = self.forward(X)[:,0,0]).log_prob(Y)



class GFLayer(nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.F1 = ISAB(embedding_dim // 2, embedding_dim // 2, 1, num_inds = 4)
        self.F2 = ISAB(embedding_dim // 2, embedding_dim // 2, 1, num_inds = 4)
        self.G1 = ISAB(embedding_dim // 2, embedding_dim // 2, 1, num_inds = 4)
        self.G2 = ISAB(embedding_dim // 2, embedding_dim // 2, 1, num_inds = 4)

    def forward(self, X):
        """
        Given X, returns Z and the log-determinant log|df/dx|.
        """
        H0 = X[:,:,:self.embedding_dim // 2]
        H1 = X[:,:,self.embedding_dim // 2:]
        F1 = self.F1(H1)
        H0 = H0 * torch.exp(F1) + self.F2(H1)
        G1 = self.G1(H0)
        H1 = H1 * torch.exp(G1) + self.G2(H0)
        logdet = torch.sum(F1, dim = 2) + torch.sum(G1, dim = 2)
        return torch.cat([H0, H1], dim = 2), logdet

    def backward(self, Z):
        """
        Given Z, returns X and the log-determinant log|df⁻¹/dz|.
        """
        H0 = Z[:,:,:self.embedding_dim // 2]
        H1 = Z[:,:,self.embedding_dim // 2:]
        G1 = self.G1(H0)
        H1 = (H1 - self.G2(H0)) / torch.exp(G1)
        F1 = self.F1(H1)
        H0 = (H0 - self.F2(H1)) / torch.exp(F1)
        logdet = -torch.sum(F1, dim = 2) - torch.sum(G1, dim = 2)
        return torch.cat([H0, H1], dim = 2), logdet


class GF(nn.Module):

    def __init__(self, n_nodes, embedding_dim, num_flows):
        super().__init__()
        self.n_nodes = n_nodes
        self.embedding_dim = embedding_dim
        self.prior = MultivariateNormal(torch.zeros(embedding_dim),
                                        torch.eye(embedding_dim))
        self.flows = nn.ModuleList([GFLayer(embedding_dim) \
                                    for _ in range(num_flows)])

    def forward(self, X):
        B, N, _ = X.shape
        log_det = torch.zeros(B, N)
        for flow in self.flows:
            X, LD = flow.forward(X)
            log_det += LD
        Z, prior_logprob = X, self.prior.log_prob(X)
        return Z, prior_logprob, log_det

    def backward(self, Z):
        B, N, _ = Z.shape
        for flow in self.flows[::-1]:
            Z, _ = flow.backward(Z)
        X = Z
        return X

