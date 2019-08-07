import math
import numpy as np
import scipy as sp
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.distributions import Normal, MultivariateNormal, Distribution
from gf.modules.attn import ISAB, PMA, MAB, SAB
from gf.modules.splines import unconstrained_RQS


class ActNorm(nn.Module):
    """
    ActNorm layer.
    """
    def __init__(self, dim, device):
        super().__init__()
        self.dim = dim
        self.device = device
        self.mu = nn.Parameter(torch.zeros(1, 1, dim, dtype = torch.float, device = self.device))
        self.log_sigma = nn.Parameter(torch.zeros(1, 1, dim, dtype = torch.float, device = self.device))
        self.initialized = False

    def forward(self, x):
        z = x * torch.exp(self.log_sigma) + self.mu
        log_det = torch.sum(self.log_sigma).repeat(x.shape[0]) * x.shape[1]
        return z, log_det

    def backward(self, z):
        x = (z - self.mu) / torch.exp(self.log_sigma)
        log_det = -torch.sum(self.log_sigma).repeat(z.shape[0]) * x.shape[1]
        return x, log_det


class OneByOneConv(nn.Module):
    """
    Invertible 1x1 convolution.
    [Kingma and Dhariwal, 2018.]
    """
    def __init__(self, dim, device):
        super().__init__()
        self.dim = dim
        self.device = device
        W, _ = sp.linalg.qr(np.random.randn(dim, dim))
        W = torch.tensor(W, dtype=torch.float, device = device)
        self.W = nn.Parameter(W)
        self.W_inv = None

    def forward(self, x):
        z = x @ self.W
        log_det = torch.slogdet(self.W)[1] * x.shape[1]
        log_det = log_det.repeat(x.shape[0])
        return z, log_det

    def backward(self, z):
        if self.W_inv is None:
            self.W_inv = torch.inverse(self.W)
        x = z @ self.W_inv
        log_det = -torch.slogdet(self.W)[1] * x.shape[1]
        log_det = log_det.repeat(z.shape[0])
        return x, log_det


class MLP(nn.Module):
    """
    Simple fully connected neural network.
    """
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.network(x)


class GFLayerNSF(nn.Module):
    """
    Neural spline flow, coupling layer.
    """
    def __init__(self, embedding_dim, device,
				 K = 5, B = 3, hidden_dim = 64, base_network = MLP):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.device = device
        self.K = K
        self.B = B
        self.f1 = nn.Sequential(
            base_network(embedding_dim // 2, hidden_dim, hidden_dim),
            SAB(hidden_dim, (3 * K - 1) * embedding_dim // 2, 1),
        )
        self.f2 = nn.Sequential(
            base_network(embedding_dim // 2, hidden_dim, hidden_dim),
            SAB(hidden_dim, (3 * K - 1) * embedding_dim // 2, 1),
        )
        self.conv = OneByOneConv(embedding_dim, device)
        self.actnorm = ActNorm(embedding_dim, device)

    def forward(self, x):
        batch_size = x.shape[0]
        x, log_det = self.actnorm(x)
        x, ld = self.conv(x)
        log_det += ld
        lower = x[:, :, :self.embedding_dim // 2]
        upper = x[:, :, self.embedding_dim // 2:]
        out = self.f1(lower).reshape(batch_size, -1, self.embedding_dim // 2,
                                     3 * self.K - 1)
        W, H, D = torch.split(out, self.K, dim = 3)
        W, H = torch.softmax(W, dim = 3), torch.softmax(H, dim = 3)
        W, H = 2 * self.B * W, 2 * self.B * H
        D = F.softplus(D)
        upper, ld = unconstrained_RQS(
            upper, W, H, D, inverse=False, tail_bound=self.B)
        log_det += torch.sum(ld, dim = (1, 2))
        out = self.f2(upper).reshape(batch_size, -1, self.embedding_dim // 2,
                                     3 * self.K - 1)
        W, H, D = torch.split(out, self.K, dim = 3)
        W, H = torch.softmax(W, dim = 3), torch.softmax(H, dim = 3)
        W, H = 2 * self.B * W, 2 * self.B * H
        D = F.softplus(D)
        lower, ld = unconstrained_RQS(
            lower, W, H, D, inverse=False, tail_bound=self.B)
        log_det += torch.sum(ld, dim = (1, 2))
        return torch.cat([lower, upper], dim = 2), log_det

    def backward(self, z):
        batch_size = z.shape[0]
        log_det = torch.zeros_like(z)
        lower = z[:, :, :self.embedding_dim // 2]
        upper = z[:, :, self.embedding_dim // 2:]
        out = self.f2(upper).reshape(batch_size, -1, self.embedding_dim // 2,
                                     3 * self.K - 1)
        W, H, D = torch.split(out, self.K, dim = 3)
        W, H = torch.softmax(W, dim = 3), torch.softmax(H, dim = 3)
        W, H = 2 * self.B * W, 2 * self.B * H
        D = F.softplus(D)
        lower, ld = unconstrained_RQS(
            lower, W, H, D, inverse=True, tail_bound=self.B)
        log_det += torch.sum(ld, dim = (1, 2))
        out = self.f1(lower).reshape(batch_size, -1, self.embedding_dim // 2,
                                     3 * self.K - 1)
        W, H, D = torch.split(out, self.K, dim = 3)
        W, H = torch.softmax(W, dim = 3), torch.softmax(H, dim = 3)
        W, H = 2 * self.B * W, 2 * self.B * H
        D = F.softplus(D)
        upper, ld = unconstrained_RQS(
            upper, W, H, D, inverse=True, tail_bound=self.B)
        log_det += torch.sum(ld, dim = (1, 2))
        x, ld1 = self.conv.backward(torch.cat([lower, upper], dim = 2))
        x, ld2 = self.actnorm.backward(x)
        return x, log_det + ld1 + ld2


class GFLayerMAF(nn.Module):
    """
    Masked auto-regressive flow style layer.
    """
    def __init__(self, embedding_dim, device):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.attn_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        self.initial_param = nn.Parameter(torch.Tensor(2))
        self.device = device
        self.conv = OneByOneConv(embedding_dim, device)
        for i in range(1, embedding_dim):
            self.fc_layers += [MLP(i, 16, hidden_dim = 128)]
            self.attn_layers += [SAB(16, 2, num_heads = 1)]
        init.uniform_(self.initial_param, -math.sqrt(0.5), math.sqrt(0.5))

    def forward(self, X):
        """
        Given X, returns Z and the log-determinant log|df/dx|.
        Note there is no reversal step because we use 1x1 conv.
        """
        Z = torch.zeros_like(X)
        Z, conv_logdet = self.conv(X)
        logdet = torch.zeros(X.shape[0], device=self.device)
        for i in range(self.embedding_dim):
            if i == 0:
                mu, alpha = self.initial_param[0], self.initial_param[1]
                alpha = alpha.repeat(X.shape[0], X.shape[1])
            else:
                out = self.attn_layers[i-1](self.fc_layers[i - 1](X[:, :, :i]))
                mu, alpha = out[:,:,0], out[:, :, 1]
            Z[:,:,i] = (X[:,:,i] - mu) / torch.exp(alpha)
            logdet -= torch.sum(alpha, dim = 1)
        #Z = Z.flip(dims=(2,))
        return Z, logdet + conv_logdet

    def backward(self, Z):
        """
        Given Z, returns X and the log-determinant log|df⁻¹/dz|.
        Note there is no reversal step because we use 1x1 conv.
        """
        X = torch.zeros_like(Z)
        #Z = Z.flip(dims=(2,))
        logdet = torch.zeros(X.shape[0], device=self.device)
        for i in range(self.embedding_dim):
            if i == 0:
                mu, alpha = self.initial_param[0], self.initial_param[1]
                alpha = alpha.repeat(X.shape[0], X.shape[1])
            else:
                out = self.attn_layers[i - 1](self.fc_layers[i - 1](X[:, :, :i]))
                mu, alpha = out[:,:,0], out[:, :, 1]
            X[:,:,i] = mu + torch.exp(alpha) * Z[:,:,i]
            logdet += alpha.sum(dim = 1)
        X, conv_logdet = self.conv.backward(X)
        return X, logdet + conv_logdet


class GFLayerNVP(nn.Module):

    def __init__(self, embedding_dim, device):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.device = device
        self.F1 = ISAB(embedding_dim // 2, embedding_dim // 2, 1, num_inds = 4)
        self.F2 = ISAB(embedding_dim // 2, embedding_dim // 2, 1, num_inds = 4)
        self.G1 = ISAB(embedding_dim // 2, embedding_dim // 2, 1, num_inds = 4)
        self.G2 = ISAB(embedding_dim // 2, embedding_dim // 2, 1, num_inds = 4)
        self.conv = OneByOneConv(embedding_dim, device)

    def forward(self, X, use_checkerboard=False):
        """
        Given X, returns Z and the log-determinant log|df/dx|.
        """
        X, logdet = self.conv(X)
        if use_checkerboard:
            H0 = X[:,:,::2]
            H1 = X[:,:,1::2]
        else:
            H0 = X[:,:,:self.embedding_dim // 2]
            H1 = X[:,:,self.embedding_dim // 2:]
        F1 = self.F1(H1)
        H0 = H0 * torch.exp(F1) + self.F2(H1)
        G1 = self.G1(H0)
        H1 = H1 * torch.exp(G1) + self.G2(H0)
        logdet += torch.sum(F1, dim = 2) + torch.sum(G1, dim = 2)
        if use_checkerboard:
            Z = torch.zeros_like(X)
            Z[:,:,::2] = H0
            Z[:,:,1::2] = H1
        else:
            Z = torch.cat([H0, H1], dim = 2)
        return Z, logdet

    def backward(self, Z, use_checkerboard=False):
        """
        Given Z, returns X and the log-determinant log|df⁻¹/dz|.
        """
        if use_checkerboard:
            H0 = Z[:,:,::2]
            H1 = Z[:,:,1::2]
        else:
            H0 = Z[:,:,:self.embedding_dim // 2]
            H1 = Z[:,:,self.embedding_dim // 2:]
        G1 = self.G1(H0)
        H1 = (H1 - self.G2(H0)) / torch.exp(G1)
        F1 = self.F1(H1)
        H0 = (H0 - self.F2(H1)) / torch.exp(F1)
        logdet = -torch.sum(F1, dim = 2) - torch.sum(G1, dim = 2)
        if use_checkerboard:
            X = torch.zeros_like(Z)
            X[:,:,::2] = H0
            X[:,:,1::2] = H1
        else:
            X = torch.cat([H0, H1], dim = 2)
        X, logdet_conv = self.conv.backward(X)
        return X, logdet + logdet_conv


class GF(nn.Module):

    def __init__(self, embedding_dim, num_flows, device):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.flows = nn.ModuleList([GFLayerNSF(embedding_dim, device) \
                                    for _ in range(num_flows)])
        self.device = device
        self.n_nodes_mu = nn.Parameter(torch.zeros(1, device = device))
        self.n_nodes_log_sigma = nn.Parameter(torch.zeros(1, device = device))

    def sample_prior(self, n_batch):
        n_nodes = 10
        prior = Normal(loc = torch.zeros(n_nodes * self.embedding_dim, 
                                         device = self.device),
                       scale = torch.ones(n_nodes * self.embedding_dim, 
                                          device = self.device))
        Z = prior.sample((n_batch,))
        Z = Z.reshape((n_batch, n_nodes, self.embedding_dim))
        return Z

    def forward(self, X):
        """
        Returns:
            log probability per node
        """
        batch_size, n_nodes = X.shape[0], X.shape[1]
        log_det = torch.zeros(batch_size, device=self.device)
        for flow in self.flows:
            X, LD = flow.forward(X)
            log_det += LD
            prior = Normal(loc = torch.zeros(n_nodes * self.embedding_dim, 
                                         device = self.device),
                       scale = torch.ones(n_nodes * self.embedding_dim, 
                                          device = self.device))
        Z, prior_logprob = X, prior.log_prob(X.view(batch_size, -1))
        prior_logprob = torch.sum(prior_logprob, dim = 1) / n_nodes
        return Z, prior_logprob + log_det

    def backward(self, Z):
        B, N, _ = Z.shape
        for flow in self.flows[::-1]:
            Z, _ = flow.backward(Z)
        return Z

