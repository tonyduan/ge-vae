import torch
import torch.nn as nn


class GFLayerMAF(nn.Module):
    """
    Masked auto-regressive flow style layer.
    """
    def __init__(self, embedding_dim, device):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.attn_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        self.initial_param = nn.Parameter(torch.Tensor(2))
        self.device = device
        self.conv = OneByOneConv(embedding_dim, device)
        for i in range(1, embedding_dim):
            self.fc_layers += [MLP(i, 16, hidden_dim = 128)]
            self.attn_layers += [SAB(16, 2, num_heads = 1)]
        init.uniform_(self.initial_param, -math.sqrt(0.5), math.sqrt(0.5))

    def forward(self, X):
        """
        Given X, returns Z and the log-determinant log|df/dx|.
        Note there is no reversal step because we use 1x1 conv.
        """
        Z = torch.zeros_like(X)
        Z, conv_logdet = self.conv(X)
        logdet = torch.zeros(X.shape[0], device=self.device)
        for i in range(self.embedding_dim):
            if i == 0:
                mu, alpha = self.initial_param[0], self.initial_param[1]
                alpha = alpha.repeat(X.shape[0], X.shape[1])
            else:
                out = self.attn_layers[i-1](self.fc_layers[i - 1](X[:, :, :i]))
                mu, alpha = out[:,:,0], out[:, :, 1]
            Z[:,:,i] = (X[:,:,i] - mu) / torch.exp(alpha)
            logdet -= torch.sum(alpha, dim = 1)
        #Z = Z.flip(dims=(2,))
        return Z, logdet + conv_logdet

    def backward(self, Z):
        """
        Given Z, returns X and the log-determinant log|df⁻¹/dz|.
        Note there is no reversal step because we use 1x1 conv.
        """
        X = torch.zeros_like(Z)
        #Z = Z.flip(dims=(2,))
        logdet = torch.zeros(X.shape[0], device=self.device)
        for i in range(self.embedding_dim):
            if i == 0:
                mu, alpha = self.initial_param[0], self.initial_param[1]
                alpha = alpha.repeat(X.shape[0], X.shape[1])
            else:
                out = self.attn_layers[i - 1](self.fc_layers[i - 1](X[:, :, :i]))
                mu, alpha = out[:,:,0], out[:, :, 1]
            X[:,:,i] = mu + torch.exp(alpha) * Z[:,:,i]
            logdet += alpha.sum(dim = 1)
        X, conv_logdet = self.conv.backward(X)
        return X, logdet + conv_logdet


class GFLayerNVP(nn.Module):

    def __init__(self, embedding_dim, device):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.device = device
        self.F1 = ISAB(embedding_dim // 2, embedding_dim // 2, 1, num_inds = 4)
        self.F2 = ISAB(embedding_dim // 2, embedding_dim // 2, 1, num_inds = 4)
        self.G1 = ISAB(embedding_dim // 2, embedding_dim // 2, 1, num_inds = 4)
        self.G2 = ISAB(embedding_dim // 2, embedding_dim // 2, 1, num_inds = 4)
        self.conv = OneByOneConv(embedding_dim, device)

    def forward(self, X, use_checkerboard=False):
        """
        Given X, returns Z and the log-determinant log|df/dx|.
        """
        X, logdet = self.conv(X)
        if use_checkerboard:
            H0 = X[:,:,::2]
            H1 = X[:,:,1::2]
        else:
            H0 = X[:,:,:self.embedding_dim // 2]
            H1 = X[:,:,self.embedding_dim // 2:]
        F1 = self.F1(H1)
        H0 = H0 * torch.exp(F1) + self.F2(H1)
        G1 = self.G1(H0)
        H1 = H1 * torch.exp(G1) + self.G2(H0)
        logdet += torch.sum(F1, dim = 2) + torch.sum(G1, dim = 2)
        if use_checkerboard:
            Z = torch.zeros_like(X)
            Z[:,:,::2] = H0
            Z[:,:,1::2] = H1
        else:
            Z = torch.cat([H0, H1], dim = 2)
        return Z, logdet

    def backward(self, Z, use_checkerboard=False):
        """
        Given Z, returns X and the log-determinant log|df⁻¹/dz|.
        """
        if use_checkerboard:
            H0 = Z[:,:,::2]
            H1 = Z[:,:,1::2]
        else:
            H0 = Z[:,:,:self.embedding_dim // 2]
            H1 = Z[:,:,self.embedding_dim // 2:]
        G1 = self.G1(H0)
        H1 = (H1 - self.G2(H0)) / torch.exp(G1)
        F1 = self.F1(H1)
        H0 = (H0 - self.F2(H1)) / torch.exp(F1)
        logdet = -torch.sum(F1, dim = 2) - torch.sum(G1, dim = 2)
        if use_checkerboard:
            X = torch.zeros_like(Z)
            X[:,:,::2] = H0
            X[:,:,1::2] = H1
        else:
            X = torch.cat([H0, H1], dim = 2)
        X, logdet_conv = self.conv.backward(X)
        return X, logdet + logdet_conv



