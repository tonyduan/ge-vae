import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from graphflows.attn import MAB, PMA
from graphflows.sparsemax import Sparsemax
from torch.distributions import MultivariateNormal, Bernoulli



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


class AttnBlock(nn.Module):

    def __init__(self, dim_Q, dim_K, dim_V, num_heads=6):
        super().__init__()
        self.dim_Q = dim_Q
        self.dim_K = dim_K
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_Q = nn.Linear(dim_Q, dim_V)
        self.fc_K = nn.Linear(dim_K, dim_V)
        self.fc_V = nn.Linear(dim_K, dim_V)
        self.fc_O = nn.Linear(dim_V, dim_V)
        self.sparsemax = Sparsemax(dim = 2)

    def forward(self, Q, V, A):
        B, _, _ = Q.shape
        Q, K, V = self.fc_Q(Q), self.fc_K(V), self.fc_V(V)
        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, dim=2), dim=0)
        K_ = torch.cat(K.split(dim_split, dim=2), dim=0)
        V_ = torch.cat(V.split(dim_split, dim=2), dim=0)
        logits = (Q_ @ K_.transpose(1, 2) / self.dim_V ** 0.5) + (1 - A) * -10 ** 10
        O = self.sparsemax(logits)
        O = torch.cat(O.split(B,), dim=2)
        return O


class MessagePassingLayer(nn.Module):

    def __init__(self, hidden_dim, message_dim, attn_layer):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.message_dim = message_dim
        self.message_layer = nn.Linear(hidden_dim, message_dim)
        self.update_layer = nn.GRUCell(message_dim, hidden_dim)
        self.attn_layer = attn_layer
        self.reset_parameters()

    def forward(self, X, A):
        B, N, _ = X.shape  # batch size, number of nodes
        M = torch.tanh(self.message_layer(X))
        W = self.attn_layer(X, X, A) # softmax weights
        M = W @ M
        M = M.view(B * N, self.message_dim)
        X = X.view(B * N, self.hidden_dim)
        X = self.update_layer(M, X).view(B, N, self.hidden_dim)
        self.attn_weights = W
        return X

    def reset_parameters(self):
        init.uniform_(self.message_layer.weight,
                      -math.sqrt(1/self.hidden_dim),
                       math.sqrt(1/self.hidden_dim))
        init.uniform_(self.message_layer.bias,
                      -math.sqrt(1/self.hidden_dim),
                       math.sqrt(1/self.hidden_dim))


class MessagePassingModule(nn.Module):

    def __init__(self, hidden_dim, message_dim, num_steps, attn_layer):
        super().__init__()
        self.num_steps = num_steps
        self.mpl = MessagePassingLayer(hidden_dim, message_dim, attn_layer)

    def forward(self, X, A):
        for _ in range(self.num_steps):
            X = self.mpl(X, A)
        return X


class GRevNetLayer(nn.Module):

    def __init__(self, hidden_dim, message_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.message_dim = message_dim
        self.attn_layer = AttnBlock(hidden_dim // 2, hidden_dim // 2, hidden_dim // 2, num_heads=1)
        self.F1 = MessagePassingModule(hidden_dim // 2, message_dim, 5, self.attn_layer)
        self.F2 = MessagePassingModule(hidden_dim // 2, message_dim, 5, self.attn_layer)
        self.G1 = MessagePassingModule(hidden_dim // 2, message_dim, 5, self.attn_layer)
        self.G2 = MessagePassingModule(hidden_dim // 2, message_dim, 5, self.attn_layer)

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

    def backward(self, Z, A):
        B, N, _ = Z.shape
        for flow in self.flows[::-1]:
            Z, _ = flow.backward(Z, A)
        X = Z
        return X

    def sample(self, n_samples):
        Z = self.prior.sample((n_samples, 18))
        A = 1 # hacky adjacency matrix construction
        x = self.backward(z)
        return x


class GAE(nn.Module):

    def __init__(self, hidden_dim, message_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.message_dim = message_dim
        self.mp_steps = nn.ModuleList([
            MessagePassingModule(hidden_dim, message_dim, 5),
            MessagePassingModule(hidden_dim, message_dim, 5),
            MessagePassingModule(hidden_dim, message_dim, 5),
            MessagePassingModule(hidden_dim, message_dim, 5),
            MessagePassingModule(hidden_dim, message_dim, 5),
        ])

    def encode(self, X, A):
        for mp in self.mp_steps:
            X = mp(X, A)
        return X

    def decode(self, X):
        pdists = torch.stack([torch.cdist(x_i, x_i) \
                             for x_i in torch.unbind(X, dim=0)], dim=0)
        A = torch.sigmoid(-10 * (pdists - 1))
        return A

    def loss(self, X, A):
        A_hat = self.decode(self.encode(X, A))
        return -Bernoulli(A_hat).log_prob(A).mean(dim=(1, 2))
