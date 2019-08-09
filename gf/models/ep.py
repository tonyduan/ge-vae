import torch.nn as nn
from torch.distributions import Bernoulli
from gf.modules.attn import MAB, PMA, SAB


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
        logits = self.forward(X).squeeze(2).squeeze(1)
        return -Bernoulli(logits = logits).log_prob(Y)


class EdgePredictor(nn.Module):

    def __init__(self, embedding_dim, device):
        super().__init__()
        self.pos_query = nn.Sequential(
            SAB(embedding_dim, embedding_dim, num_heads = 1),
            SAB(embedding_dim, embedding_dim, num_heads = 1),
            SAB(embedding_dim, embedding_dim, num_heads = 1)
        )
        self.device = device

    def forward(self, E):
        Z = self.pos_query(E)
        return Z @ Z.transpose(1, 2)

    def loss(self, E, A):
        logits = self.forward(E)
        return -Bernoulli(logits = logits).log_prob(A)

