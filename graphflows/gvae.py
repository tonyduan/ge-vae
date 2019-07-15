import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.distributions import Normal, Bernoulli, kl


class DiagGCN(nn.Module):

    def __init__(self, n_nodes, latent_dim):
        super().__init__()
        self.W0 = nn.Parameter(torch.Tensor(n_nodes, n_nodes))
        self.W1_mu = nn.Parameter(torch.Tensor(n_nodes, latent_dim))
        self.W1_sd = nn.Parameter(torch.Tensor(n_nodes, latent_dim))
        self.reset_parameters(n_nodes)

    def reset_parameters(self, dim):
        init.uniform_(self.W0, -math.sqrt(1/dim), math.sqrt(1/dim))
        init.uniform_(self.W1_mu, -math.sqrt(1/dim), math.sqrt(1/dim))
        init.uniform_(self.W1_sd, -math.sqrt(1/dim), math.sqrt(1/dim))

    def forward(self, A):
        D = torch.sum(A, dim=1)
        D = torch.diag_embed(1 / torch.pow(D, 0.5))
        A_tilde = D @ A @ D
        mu = A_tilde @ torch.relu(A_tilde @ self.W0) @ self.W1_mu
        sd = A_tilde @ torch.relu(A_tilde @ self.W0) @ self.W1_sd
        return Normal(loc=mu, scale=torch.exp(sd))


class GVAE(nn.Module):

    def __init__(self, n_nodes, latent_dim):
        super().__init__()
        self.n_nodes = n_nodes
        self.latent_dim = latent_dim
        self.prior = Normal(torch.zeros(latent_dim), torch.ones(latent_dim))
        self.encoder = DiagGCN(n_nodes, latent_dim)

    def loss(self, A):
        pred_z = self.compute_z_given_a(A)
        kl_div = kl.kl_divergence(pred_z, self.prior).squeeze(1)
        monte_carlo_z = pred_z.rsample()
        monte_carlo_a = self.compute_a_given_z(monte_carlo_z)
        rec_loss = -torch.sum(monte_carlo_a.log_prob(A), dim=1)
        return kl_div.sum(dim=2).sum(dim=1) + rec_loss.sum(dim=1)

    def sample_z(self, n_samples):
        return self.prior.rsample((n_samples, self.n_nodes))

    def compute_a_given_z(self, Z):
        bernoulli_matrix = torch.sigmoid(Z @ torch.transpose(Z, 1, 2))
        return Bernoulli(bernoulli_matrix)

    def compute_z_given_a(self, A):
        return self.encoder(A)
