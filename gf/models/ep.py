import torch
import torch.nn as nn
from torch.distributions import Bernoulli
from gf.modules.attn import MAB, PMA, SAB, ISAB, ISABStack
from gf.utils import *


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
        self.pos_query = ISABStack(3, embedding_dim, 128, num_heads = 4, num_inds = 16, device = device)
        self.neg_query = ISABStack(3, embedding_dim, 128, num_heads = 4, num_inds = 16, device = device)
        self.gate = ISABStack(3, embedding_dim, 128, num_heads = 4, num_inds = 16, device = device)
        self.device = device
        self.foc = 10.0

    def forward(self, E, V):
        batch_size, max_n_nodes = E.shape[0], E.shape[1]
        mask = construct_embedding_mask(E, V).byte()
        G = torch.sigmoid(self.gate(E, mask))
        Z = self.pos_query(E, mask) * G + self.neg_query(E, mask) * (1 - G)
        return Z @ Z.transpose(1, 2) 

    def loss(self, E, A, V):
        batch_size, max_n_nodes = E.shape[0], E.shape[1]
        logits = self.forward(E, V)
        bern = Bernoulli(logits = logits)
        denom = V * (V - 1) / 2
        mask = construct_adjacency_mask(E, V)
        loss = bern.log_prob(A) * (1 - torch.exp(bern.log_prob(A)) ** self.foc)
        return -torch.sum(torch.triu(loss, diagonal = 1) * mask, dim = (1, 2)) / denom

