import torch
import torch.nn as nn
from torch.distributions import Bernoulli
from gf.modules.attn import MAB, PMA, SAB, ISAB, ISABStack 
from gf.utils import *



class MLP(nn.Module):
    """
    Simple fully connected neural network.
    """
    def __init__(self, in_dim, out_dim, hidden_dim, device):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        ).to(device)

    def forward(self, x):
        return self.network(x)
#
#class EdgePredictor(nn.Module):
#
#    def __init__(self, embedding_dim, device = "cpu"):
#        super().__init__()
#        self.rest_query = MAB(embedding_dim, embedding_dim, embedding_dim, 1)
#        self.pair_query = PMA(embedding_dim, num_heads = 1, num_seeds = 1)
#        self.fc = nn.Linear(embedding_dim, 1)
#
#    def forward(self, X):
#        pair = X[:, :2, :]
#        rest = X[:, 2:, :]
#        return self.fc(self.pair_query(self.rest_query(pair, rest)))
#
#    def loss(self, X, Y):
#        logits = self.forward(X).squeeze(2).squeeze(1)
#        return -Bernoulli(logits = logits).log_prob(Y)
#
#

class EdgePredictor(nn.Module):

    def __init__(self, embedding_dim, device):
        super().__init__()
        self.query = ISABStack(2, embedding_dim, 256, num_heads = 4, num_inds = 16, device = device)
        self.query2 = ISABStack(2, embedding_dim, 128, num_heads = 4, num_inds = 16, device = device)
        #self.query3 = ISABStack(2, embedding_dim, 1, num_heads = 4, num_inds = 16, device = device)
        #self.query = MLP(embedding_dim, 128, 128, device)
        self.mlp = MLP(128, 1, 128, device)
        self.query3 = MLP(embedding_dim, 1, 128, device)
        self.device = device
        self.baseline = nn.Parameter(torch.zeros(1, device = device))
        self.baseline2 = nn.Parameter(torch.zeros(1, device = device))
        self.baseline3 = nn.Parameter(torch.zeros(1, device = device))
        self.baseline4 = nn.Parameter(torch.zeros(1, device = device))
        self.baseline5 = nn.Parameter(torch.zeros(1, device = device))
        self.foc = 0.0

    def forward(self, E, V):
        mask = construct_embedding_mask(V).byte()
        #G = torch.sigmoid(self.gate(E, mask))
        #Z = self.pos_query(E, mask) * G + self.neg_query(E, mask) * (1 - G) 
        Z = self.query(E, mask)
        F = Z @ Z.transpose(1, 2)
        Z2 = self.mlp(self.query2(E, mask))
        G = Z2 @ Z2.transpose(1,2)
        Z3 = self.query3(E.mean(dim = 1)).unsqueeze(2)
        H = Z3 @ Z3.transpose(1, 2)
        return F * torch.exp(self.baseline2) + \
               G * torch.exp(self.baseline3) + \
               H * torch.exp(self.baseline4) + \
               self.baseline

    def log_prob_per_edge(self, E, A, V):
        logits = self.forward(E, V)
        bern = Bernoulli(logits = logits)
        denom = V * (V - 1) / 2
        mask = construct_adjacency_mask(V)
        loss = bern.log_prob(A) * ((1 - torch.exp(bern.log_prob(A))) ** self.foc)
        return torch.sum(torch.triu(loss, diagonal = 1) * mask, dim = (1, 2)) / denom

