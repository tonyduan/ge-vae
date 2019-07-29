import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.distributions import Normal, Bernoulli, MultivariateNormal
from graphflows.attn import ISAB, PMA, MAB, SAB


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


class MLP(nn.Module):
    """
    Simple fully connected neural network.
    """
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.network(x)

class GFLayer(nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.attn_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        self.initial_param = nn.Parameter(torch.Tensor(2))
        for i in range(1, embedding_dim):
            self.attn_layers += [SAB(i, i, num_heads = 1)]
            self.fc_layers += [nn.Linear(i, 2)]
            # self.fc_layers += [MLP(i, 2, hidden_dim = 16)]

    def forward(self, X):
        """
        Given X, returns Z and the log-determinant log|df/dx|.
        """
        Z = torch.zeros_like(X)
        logdet = torch.zeros(X.shape[0], X.shape[1])
        for i in range(self.embedding_dim):
            if i == 0:
                mu, alpha = self.initial_param[0], self.initial_param[1]
            else:
                out = self.attn_layers[i - 1](X[:, :, :i])
                out = self.fc_layers[i -1](out)
                mu, alpha = out[:,:,0], out[:, :, 1]
            Z[:,:,i] = (X[:,:,i] - mu) / torch.exp(alpha)
            logdet -= alpha
        return Z, logdet

    def backward(self, Z):
        """
        Given Z, returns X and the log-determinant log|df⁻¹/dz|.
        """
        X = torch.zeros_like(Z)
        logdet = torch.zeros(X.shape[0], X.shape[1])
        for i in range(self.embedding_dim):
            if i == 0:
                mu, alpha = self.initial_param[0], self.initial_param[1]
            else:
                out = self.fc_layers[i -1](self.attn_layers[i - 1](X[:, :, :i]))
                mu, alpha = out[:,:,0], out[:, :, 1]
            X[:,:,i] = mu + torch.exp(alpha) * Z[:,:,i]
            logdet += alpha
        return X, logdet


class GFLayerNVP(nn.Module):

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
        X = X.flip((2,))
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
        X = X.flip((2,))
        return X

