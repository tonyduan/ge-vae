import math
import numpy as np
import scipy as sp
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.distributions import *
from gf.modules.attn import *
from gf.modules.splines import unconstrained_RQS
from gf.modules.mlp import MLP
from gf.models.ep import EdgePredictor
from gf.utils import *


class ActNorm(nn.Module):
    """
    ActNorm layer.
    """
    def __init__(self, dim, device):
        super().__init__()
        self.dim = dim
        self.device = device
        self.mu = nn.Parameter(torch.zeros(1, 1, dim, 
            dtype = torch.float, device = self.device))
        self.log_sigma = nn.Parameter(torch.zeros(1, 1, dim, 
            dtype = torch.float, device = self.device))
        self.initialized = False

    def forward(self, x, v):
        z = x * torch.exp(self.log_sigma) + self.mu
        log_det = torch.sum(self.log_sigma).repeat(x.shape[0]) * v
        return z, log_det

    def backward(self, z, v):
        x = (z - self.mu) / torch.exp(self.log_sigma)
        log_det = -torch.sum(self.log_sigma).repeat(z.shape[0]) * v
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

    def forward(self, x, v):
        z = x @ self.W
        log_det = torch.slogdet(self.W)[1].repeat(x.shape[0]) * v
        return z, log_det

    def backward(self, z, v):
        if self.W_inv is None:
            self.W_inv = torch.inverse(self.W)
        x = z @ self.W_inv
        log_det = -torch.slogdet(self.W)[1].repeat(z.shape[0]) * v
        return x, log_det


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
#        self.f1 = ISAB(embedding_dim // 2, hidden_dim, 1, 16)
#        self.f2 = ISAB(embedding_dim // 2, hidden_dim, 1, 16)
        self.f1 = ISABStack(1, embedding_dim // 2, hidden_dim, 1, 16)
        self.f2 = ISABStack(1, embedding_dim // 2, hidden_dim, 1, 16)
        self.base_network = base_network(
            hidden_dim, (3 * K - 1) * embedding_dim // 2, hidden_dim, device)
        self.conv = OneByOneConv(embedding_dim, device)
        self.actnorm = ActNorm(embedding_dim, device)

    def forward(self, x, v):
        batch_size, max_n_nodes = x.shape[0], x.shape[1]
        mask = construct_embedding_mask(v)
        x, log_det = self.actnorm(x, v)
        x, ld = self.conv(x, v)
        log_det += ld
        lower = x[:, :, :self.embedding_dim // 2]
        upper = x[:, :, self.embedding_dim // 2:]
        out = self.base_network(self.f1(lower, mask.byte())).reshape(
            batch_size, -1, self.embedding_dim // 2, 3 * self.K - 1)
        W, H, D = torch.split(out, self.K, dim = 3)
        W, H = torch.softmax(W, dim = 3), torch.softmax(H, dim = 3)
        W, H = 2 * self.B * W, 2 * self.B * H
        D = F.softplus(D)
        upper, ld = unconstrained_RQS(
            upper, W, H, D, inverse=False, tail_bound=self.B)
        log_det += torch.sum(ld * mask.unsqueeze(2), dim = (1, 2))
        out = self.base_network(self.f2(upper, mask.byte())).reshape(
            batch_size, -1, self.embedding_dim // 2, 3 * self.K - 1)
        W, H, D = torch.split(out, self.K, dim = 3)
        W, H = torch.softmax(W, dim = 3), torch.softmax(H, dim = 3)
        W, H = 2 * self.B * W, 2 * self.B * H
        D = F.softplus(D)
        lower, ld = unconstrained_RQS(
            lower, W, H, D, inverse=False, tail_bound=self.B)
        log_det += torch.sum(ld * mask.unsqueeze(2), dim = (1, 2))
        return torch.cat([lower, upper], dim = 2), log_det

    def backward(self, z, v):
        batch_size, max_n_nodes = z.shape[0], z.shape[1]
        mask = construct_embedding_mask(v)
        log_det = torch.zeros_like(v)
        lower = z[:, :, :self.embedding_dim // 2]
        upper = z[:, :, self.embedding_dim // 2:]
        out = self.base_network(self.f2(upper, mask.byte())).reshape(
            batch_size, -1, self.embedding_dim // 2,  3 * self.K - 1)
        W, H, D = torch.split(out, self.K, dim = 3)
        W, H = torch.softmax(W, dim = 3), torch.softmax(H, dim = 3)
        W, H = 2 * self.B * W, 2 * self.B * H
        D = F.softplus(D)
        lower, ld = unconstrained_RQS(
            lower, W, H, D, inverse=True, tail_bound=self.B)
        log_det += torch.sum(ld * mask.unsqueeze(2), dim = (1, 2))
        out = self.base_network(self.f1(lower, mask.byte())).reshape(
            batch_size, -1, self.embedding_dim // 2, 3 * self.K - 1)
        W, H, D = torch.split(out, self.K, dim = 3)
        W, H = torch.softmax(W, dim = 3), torch.softmax(H, dim = 3)
        W, H = 2 * self.B * W, 2 * self.B * H
        D = F.softplus(D)
        upper, ld = unconstrained_RQS(
            upper, W, H, D, inverse=True, tail_bound=self.B)
        log_det += torch.sum(ld *  mask.unsqueeze(2), dim = (1, 2))
        x, ld1 = self.conv.backward(torch.cat([lower, upper], dim = 2), v)
        x, ld2 = self.actnorm.backward(x, v)
        return x, log_det + ld1 + ld2

class GF(nn.Module):

    def __init__(self, embedding_dim, num_flows, device):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.flows_L = nn.ModuleList([GFLayerNSF(embedding_dim, device) \
                                      for _ in range(num_flows)])
        self.flows_Z = nn.ModuleList([GFLayerNSF(embedding_dim, device) \
                                      for _ in range(num_flows)])
        self.ep = EdgePredictor(embedding_dim, device)
        self.final_actnorm = ActNorm(embedding_dim, device)
        self.prior_gen = lambda n_nodes: \
            Normal(torch.zeros(embedding_dim * n_nodes, device = self.device),
                   torch.ones(embedding_dim * n_nodes, device = self.device))
        self.device = device

    def sample_prior(self, n_batch, n_nodes = 20):
        z = self.prior_gen(n_nodes).sample((n_batch,))
        z = z.reshape((n_batch, n_nodes, self.embedding_dim))
        v = torch.ones(n_batch) * n_nodes
        return z, v

    def forward(self, x, a, v):
        """
        Returns
        -------
        z: (batch_size) x (max_n_nodes) x (embedding_size) latent features

        lpn: (batch_size) log probability per node

        lpe: (batch_size) log probability per edge
        """
        batch_size, max_n_nodes = x.shape[0], x.shape[1]
        log_det = torch.zeros(batch_size, device=self.device)
        for flow in self.flows_L:
            x, LD = flow.forward(x, v)
            log_det += LD
        ep_logprob = self.ep.log_prob_per_edge(x, a, v)
        for flow in self.flows_Z:
            x, LD = flow.forward(x, v)
            log_det += LD
        x, LD = self.final_actnorm(x, v)
        log_det += LD
        prior = self.prior_gen(max_n_nodes)
        z, prior_logprob = x, prior.log_prob(x.view(batch_size, -1))
        prior_logprob = prior_logprob.reshape((batch_size, max_n_nodes, -1))
        mask = construct_embedding_mask(v)
        prior_logprob = torch.sum(prior_logprob * mask.unsqueeze(2), dim = (1, 2))
        return z, (prior_logprob + log_det) / v / self.embedding_dim, ep_logprob 

    def predict_a_from_e(self, X, V):
        batch_size, n_nodes = X.shape[0], X.shape[1]
        for flow in self.flows_L:
            X,  _ = flow.forward(X, V)
        return Bernoulli(logits = self.ep.forward(X, V)).probs

    def backward(self, z, v):
        z, _ = self.final_actnorm.backward(z, v)
        for flow in self.flows_Z[::-1]:
            z, _ = flow.backward(z, v)
        return z

