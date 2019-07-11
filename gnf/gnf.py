import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal


class MessagePassingLayer(nn.Module):

    def __init__(self, hidden_dim, message_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.message_dim = message_dim
        self.message_layer = nn.Linear(hidden_dim, message_dim)
        self.update_layer = nn.GRUCell(message_dim, hidden_dim)

    def forward(self, X, A):
        B, N, _ = X.shape                         # batch size, number of nodes
        M = torch.tanh(self.message_layer(X))
        M = A @ M # simple sum for now, todo: replace with attention
        M = M.view(B * N, self.message_dim)
        X = X.view(B * N, self.hidden_dim)
        X = self.update_layer(M, X).view(B, N, self.hidden_dim)
        return X

class MessagePassingModule(nn.Module):

    def __init__(self, hidden_dim, message_dim, num_steps):
        super().__init__()
        self.num_steps = num_steps
        self.mpl = MessagePassingLayer(hidden_dim, message_dim)

    def forward(self, X, A):
        for _ in range(self.num_steps):
            X = self.mpl(X, A)
        return X

class GRevNetLayer(nn.Module):

    def __init__(self, hidden_dim, message_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.message_dim = message_dim
        self.F1 = MessagePassingModule(hidden_dim // 2, message_dim, 5)
        self.F2 = MessagePassingModule(hidden_dim // 2, message_dim, 5)
        self.G1 = MessagePassingModule(hidden_dim // 2, message_dim, 5)
        self.G2 = MessagePassingModule(hidden_dim // 2, message_dim, 5)

    def forward(self, X, A):
        """
        Given X, returns Z and the log-determinant log|df/dx|.
        """
        H0 = X[:,:,:self.hidden_dim // 2]
        H1 = X[:,:,self.hidden_dim // 2:]
        F1 = self.F1(H1, A)
        H0 = H0 * torch.exp(F1) + self.F2(H1, A)
        G1 = self.G1(H0, A)
        H1 = H1 * torch.exp(G1) + self.G2(H0, A)
        logdet = torch.sum(F1, dim = 2) + torch.sum(G1, dim = 2)
        return torch.cat([H0, H1], dim = 2), logdet

    def backward(self, Z, A):
        """
        Given Z, returns X and the log-determinant log|df⁻¹/dz|.
        """
        H0 = Z[:,:,:self.hidden_dim // 2]
        H1 = Z[:,:,self.hidden_dim // 2:]
        G1 = self.G1(H0, A)
        H1 = (H1 - self.G2(H0, A)) / torch.exp(G1)
        F1 = self.F1(H1, A)
        H0 = (H0 - self.F2(H1, A)) / torch.exp(F1)
        logdet = -torch.sum(F1, dim = 2) - torch.sum(G1, dim = 2)
        return torch.cat([H0, H1], dim = 2), logdet


class GRevNet(nn.Module):

    def __init__(self, hidden_dim, message_dim, num_layers):
        super().__init__()
        self.prior = MultivariateNormal(torch.zeros(hidden_dim),
                                        torch.eye(hidden_dim))
        self.flows = nn.ModuleList([GRevNetLayer(hidden_dim, message_dim) \
                                    for _ in range(num_layers)])

    def forward(self, X, A):
        B, N, _ = X.shape
        log_det = torch.zeros(B, N)
        for flow in self.flows:
            X, LD = flow.forward(X, A)
            log_det += LD
        Z, prior_logprob = X, self.prior.log_prob(X)
        return Z, prior_logprob, log_det

    def backward(self, z):
        pass
        
    def sample(self, n_samples):
        z = self.prior.sample((n_samples,))
        x, _, _ = self.backward(z)
        return x
