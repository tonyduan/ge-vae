import torch
import torch.nn as nn
from torch.distributions import Bernoulli
from gf.modules.attn import MAB, PMA, SAB, ISAB, ISABStack 
from gf.utils import *
from gf.modules.mlp import *


class EdgePredictor(nn.Module):

    def __init__(self, embedding_dim, device):
        super().__init__()
        self.pairwise_query = ISABStack(8, embedding_dim, 256, num_heads = 4, 
                                        num_inds = 16, device = device)
        self.device = device
        self.baseline = nn.Parameter(torch.zeros(1, device = device))
        self.scale1 = nn.Parameter(torch.zeros(1, device = device))

    def forward(self, E, V):
        mask = construct_embedding_mask(V).byte()
        Z1 = self.pairwise_query(E, mask)
        F = Z1 @ Z1.transpose(1, 2)
        return  F * torch.exp(self.scale1) + self.baseline #+ \

    def log_prob_per_edge(self, E, A, V):
        mask = construct_adjacency_mask(V)
        counts = V * (V - 1) / 2
        loss = Bernoulli(logits = self.forward(E, V)).log_prob(A)
        loss = torch.sum(torch.triu(loss, diagonal = 1) * mask, dim = (1, 2))
        return loss #/ counts

