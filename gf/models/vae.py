import torch
import torch.nn as nn
from torch.distributions import Normal, kl


class VAE(nn.Module):

    def __init__(self, input_dim, prior_dim, hidden_dim):
        super().__init__()
        self.prior = Normal(torch.zeros(prior_dim), torch.ones(prior_dim))
        self.encoder = DiagNormalNetwork(input_dim, prior_dim, hidden_dim)
        self.decoder = DiagNormalNetwork(prior_dim, input_dim, hidden_dim)

    def loss(self, x):
        pred_z = self.compute_z_given_x(x)
        kl_div = kl.kl_divergence(pred_z, self.prior).squeeze(1)#.sum(1)
        monte_carlo_z = pred_z.rsample()
        monte_carlo_x = self.compute_x_given_z(monte_carlo_z)
        rec_loss = -torch.sum(monte_carlo_x.log_prob(x), dim=1)
        return kl_div + rec_loss

    def sample_z(self, n_samples):
        return self.prior.rsample((n_samples,))

    def compute_x_given_z(self, z):
        return self.decoder(z)

    def compute_z_given_x(self, x):
        return self.encoder(x)


class InfoVAE(VAE):

    def __init__(self, input_dim, prior_dim, hidden_dim, alpha, lambd, div):
        super().__init__(input_dim, prior_dim, hidden_dim)
        self.prior_dim = prior_dim
        self.alpha = alpha
        self.lambd = lambd
        self.div = div

    def loss(self, x):
        pred_z = self.compute_z_given_x(x)
        kl_div = kl.kl_divergence(pred_z, self.prior).squeeze(1)
        monte_carlo_z = pred_z.rsample()
        monte_carlo_x = self.compute_x_given_z(monte_carlo_z)
        rec_loss = -torch.sum(monte_carlo_x.log_prob(x), dim=1)
        monte_carlo_prior = self.prior.rsample((200,))
        div = self.div(monte_carlo_prior, monte_carlo_z)
        return rec_loss + (1 - self.alpha) * kl_div + \
               (self.alpha + self.lambd - 1) * div


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
        mean, sd = torch.split(params, params.shape[1] // 2, dim=1)
        return Normal(loc=mean, scale=torch.exp(sd))
        

def compute_diffs(x, y):
    tiled_x = x.unsqueeze(dim=1).expand(x.shape[0], y.shape[0], x.shape[1])
    tiled_y = y.unsqueeze(dim=0).expand(x.shape[0], y.shape[0], y.shape[1])
    return tiled_x - tiled_y


def mmd_divergence(p_samples, q_samples):
    k_pq = compute_diffs(p_samples, q_samples)
    k_pp = compute_diffs(p_samples, p_samples)
    k_qq = compute_diffs(q_samples, q_samples)
    return torch.exp(-torch.mean(torch.pow(k_pp, 2), dim=2)).mean() + \
           torch.exp(-torch.mean(torch.pow(k_qq, 2), dim=2)).mean() - \
           2 * torch.exp(-torch.mean(torch.pow(k_pq, 2), dim=2)).mean()


