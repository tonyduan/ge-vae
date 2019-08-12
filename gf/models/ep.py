import torch
import torch.nn as nn
from torch.distributions import Bernoulli
from gf.modules.attn import MAB, PMA, SAB, ISAB


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

class EdgePredictor(nn.Module):

    def __init__(self, embedding_dim, device):
        super().__init__()
        self.rest_query = MAB(embedding_dim, embedding_dim, embedding_dim, 1)
        self.pair_query = PMA(embedding_dim, num_heads = 1, num_seeds = 1)
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, X):
        pair = X[:, :2, :]
        rest = X[:, 2:, :]
        return self.fc(self.pair_query(self.rest_query(pair, rest)))

    def loss(self, X, Y):
        logits = self.forward(X).squeeze(2).squeeze(1)
        return -Bernoulli(logits = logits).log_prob(Y)

class EdgePredictor(nn.Module):

    def __init__(self, embedding_dim, device):
        super().__init__()
        self.query = nn.Sequential(
            ISAB(embedding_dim, embedding_dim, num_heads = 1, num_inds = 8, device = device),
            ISAB(embedding_dim, embedding_dim, num_heads = 1, num_inds = 8, device = device),
            ISAB(embedding_dim, embedding_dim, num_heads = 1, num_inds = 8, device = device),
        )
        #self.gate = nn.Sequential(
        #    ISAB(embedding_dim, embedding_dim, num_heads = 1, num_inds = 8),
        #    ISAB(embedding_dim, embedding_dim, num_heads = 1, num_inds = 8),
        #    ISAB(embedding_dim, embedding_dim, num_heads = 1, num_inds = 8),
        #    nn.Sigmoid()
        #)
        #self.pos_query = nn.Sequential(
        #    ISAB(embedding_dim, embedding_dim, num_heads = 1, num_inds = 8),
        #    ISAB(embedding_dim, embedding_dim, num_heads = 1, num_inds = 8),
        #    ISAB(embedding_dim, embedding_dim, num_heads = 1, num_inds = 8),
        #)
        self.device = device

    def forward(self, E):
        #G = self.gate(E)
        #Z = self.pos_query(E) * G - (1 - G) * self.neg_query(E)
        Z = self.query(E)
        return Z @ Z.transpose(1, 2) 

    def loss(self, E, A):
        logits = self.forward(E)
        bern = Bernoulli(logits = logits)
        n_nodes = E.shape[1]
        denom = n_nodes * (n_nodes - 1) / 2
        return -torch.sum(torch.triu(bern.log_prob(A) * (1 - torch.exp(bern.log_prob(A)) ** 5), diagonal = 1), dim = (1, 2)) / denom

