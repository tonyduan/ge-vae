
import math
import numpy as np
import scipy as sp
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.distributions import Normal, MultivariateNormal, Distribution, Poisson, Bernoulli, kl
from gf.modules.attn import ISAB, PMA, MAB, SAB
from gf.modules.splines import unconstrained_RQS
from gf.models.ep import EdgePredictor
from gf.models.gnf import MessagePassingModule
from gf.utils import *


class DiagNormalNetwork(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2 * out_dim),
        )

    def forward(self, x):
        params = self.network(x)
        mean, sd = torch.split(params, params.shape[-1] // 2, dim=-1)
        return Normal(loc=mean, scale=torch.exp(sd))


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


class GFVAE(nn.Module):

    def __init__(self, embedding_dim, num_mp_steps, device):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.device = device
        self.ep = EdgePredictor(embedding_dim, device)
        self.mp_steps = nn.ModuleList([
            MessagePassingModule(embedding_dim, 128, 5) for _ in range(num_mp_steps)
        ])
        self.final_encoder = DiagNormalNetwork(embedding_dim, embedding_dim, 128)

    def sample_prior(self, n_batch):
        n_nodes = 20
        prior = Normal(loc = torch.zeros(n_nodes * self.embedding_dim, 
                                         device = self.device),
                       scale = torch.ones(n_nodes * self.embedding_dim, 
                                          device = self.device))
        Z = prior.sample((n_batch,))
        Z = Z.reshape((n_batch, n_nodes, self.embedding_dim))
        V = torch.ones(n_batch) * n_nodes
        return Z, V

    def forward(self, x, a, v):
        batch_size, max_n_nodes = x.shape[0], x.shape[1]
        mask = construct_embedding_mask(v)
        for mp in self.mp_steps:
            x = mp(x, a)
        pred_z = self.final_encoder(x)
        prior = Normal(loc = torch.zeros(max_n_nodes, self.embedding_dim, 
                                         device = self.device),
                       scale = torch.ones(max_n_nodes, self.embedding_dim, 
                                          device = self.device))
        kl_div = torch.mean(kl.kl_divergence(pred_z, prior) * mask.unsqueeze(2), dim = (1, 2)) * v
        monte_carlo_z = pred_z.sample()
        ep_logprob = self.ep.log_prob_per_edge(monte_carlo_z, a, v)
        return monte_carlo_z, -kl_div, ep_logprob 

    def predict_A_from_E(self, x, a, v):
        batch_size, n_nodes = x.shape[0], x.shape[1]
        for mp in self.mp_steps:
            x = mp(x, a)
        pred_z = self.final_encoder(x)
        monte_carlo_z = pred_z.sample()
        return Bernoulli(logits = self.ep.forward(monte_carlo_z, v)).probs

    def backward(self, Z, V):
        B, N, _ = Z.shape
        Z, _ = self.final_actnorm.backward(Z, V)
        for flow in self.flows_Z[::-1]:
            Z, _ = flow.backward(Z, V)
        return Z

