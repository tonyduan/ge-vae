import torch
from collections import defaultdict
from torch.utils.data import Dataset 
from torch.utils.data.sampler import Sampler, SequentialSampler


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

