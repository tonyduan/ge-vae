import torch
from collections import defaultdict
from torch.utils.data import Dataset 
from torch.utils.data.sampler import Sampler, SequentialSampler


def custom_collate_fn(batch):
    """
    Collate function that batches graphs of varying sizes together into the
    same mini-batch, by padding embeddings / adjacency matrices with zeros.
    Masks are tracked by returning |V| as an extra dataset.
    """
    batch_size = len(batch)
    embed_size = len(batch[0][0][0])
    max_n_nodes = max([len(l) for l, a in batch])
    L = torch.zeros(batch_size, max_n_nodes, embed_size, device = batch[0][0].device)
    A = torch.zeros(batch_size, max_n_nodes, max_n_nodes, device = batch[0][0].device)
    V = torch.zeros(batch_size, device = batch[0][0].device, dtype = torch.float)
    for i, (l, a) in enumerate(batch):
        n_nodes = len(l)
        V[i] = n_nodes
        L[i, :n_nodes, :] = l
        A[i, :n_nodes, :n_nodes] = a
    return L, A, V

class CustomBatchSampler(Sampler):
    """
    Custom batch sampler that loads the dataset sequentially, but emits a new
    batch each time a graph of different shape is encountered. Results in 
    batch sizes that re not always even.
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
    def __iter__(self):
        batch = []
        for idx in range(len(self.dataset)):
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
            if idx + 1 < len(self.dataset) and \
               len(self.dataset[idx + 1][0]) != len(self.dataset[idx][0]):
                yield batch
                batch = []
        if len(batch) > 0:
            yield batch
    def __len__(self):
        raise ValueError("Not implemented.")


class EmbeddingBatchSampler(Sampler):
    """
    Custom batch sampler that loads the dataset sequentially, but emits a new
    batch each time a graph of different shape is encountered. Results in 
    batch sizes that re not always even.
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
    def __iter__(self):
        batch = []
        for idx in range(len(self.dataset)):
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
            if idx + 1 < len(self.dataset) and \
               len(self.dataset[idx + 1]) != len(self.dataset[idx]):
                yield batch
                batch = []
        if len(batch) > 0:
            yield batch
    def __len__(self):
        raise ValueError("Not implemented.")

class EdgeDataset(Dataset):
    
    def __init__(self, X, Y, device):
        assert len(X) == len(Y)
        self.X = X
        self.Y = Y
        self.device = device

    def __getitem__(self, index):
        return torch.tensor(self.X[index], dtype = torch.float, device = self.device), \
               torch.tensor(self.Y[index], dtype = torch.float, device = self.device)

    def __len__(self):
        return len(self.X)


class EmbeddingDataset(Dataset):
    
    def __init__(self, E, device):
        self.E = E
        self.device = device

    def __getitem__(self, index):
        return torch.tensor(self.E[index], dtype = torch.float, 
                            device = self.device)

    def __len__(self):
        return len(self.E)


class GraphDataset(Dataset):

    def __init__(self, L, A, device):
        self.L = L
        self.A = A
        self.device =  device

    def __getitem__(self, index):
        return torch.tensor(self.L[index], dtype = torch.float,
                            device = self.device), \
               torch.tensor(self.A[index], dtype = torch.float,
                            device = self.device)

    def __len__(self):
        return len(self.L)



