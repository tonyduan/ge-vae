import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.distributions import Normal, Bernoulli, kl
from graphflows.attn import ISAB, PMA


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


class GraphFlow(nn.Module):

    def __init__(self, n_nodes, latent_dim):
        super().__init__()
        self.n_nodes = n_nodes
        self.latent_dim = latent_dim
        self.prior = Normal(torch.zeros(latent_dim), torch.ones(latent_dim))
        self.isab = nn.Sequential(
            ISAB(dim_in = 18, dim_out = 18, num_heads=2, num_inds=5),
        )
        self.pma1 = PMA(dim = 18, num_heads = 2, num_seeds = 1)
        self.pma2 = PMA(dim = 18, num_heads = 2, num_seeds = 1)
        self.iisab = nn.Sequential(
            PMA(dim = 18, num_heads = 2, num_seeds = 12), # number of evecs
            ISAB(dim_in = 18, dim_out = 18, num_heads = 2, num_inds=5),
        )

    def loss(self, A, E):
        pred_z = self.compute_z_given_e(E)
        kl_div = kl.kl_divergence(pred_z, self.prior)
        monte_carlo_z = pred_z.rsample()
        monte_carlo_a = self.compute_a_given_z(monte_carlo_z)
        rec_loss = -monte_carlo_a.log_prob(A) * (1-torch.eye(A.shape[1])).repeat(A.shape[0], 1, 1)
        return kl_div.mean(dim=(1, 2)) + rec_loss.mean(dim=(1, 2))

    def sample_z(self, n_samples):
        return self.prior.rsample((n_samples, self.n_nodes))

    def compute_e_given_z(self, Z):
        H = self.iisab(Z)

        return Bernoulli(bernoulli_matrix)

    def forward(self, E):
        H = self.isab(E)
        mu = self.pma1(H)
        logsd = self.pma2(H)
        return Normal(loc=mu, scale=torch.exp(logsd))
