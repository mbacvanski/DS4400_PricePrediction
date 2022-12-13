import pickle
from typing import List, Union

import numpy as np
import torch
from scipy.sparse import coo_matrix, csr_matrix
from torch import nn
from torch.nn import Linear, ReLU
from torch.utils.data import DataLoader, Dataset

with open('x_train_transformed.pkl', 'rb') as f:
    X_train = pickle.load(f)

with open('x_test_transformed.pkl', 'rb') as f:
    X_test = pickle.load(f)

with open('y_train.pkl', 'rb') as f:
    Y_train = pickle.load(f)

with open('y_test.pkl', 'rb') as f:
    Y_test = pickle.load(f)


class DNN(nn.Module):
    def __init__(self, layer_sizes: List[int]):
        """
        The first layer size is the input dimension, and the last layer size is the output dimension.
        :param layer_sizes:
        """
        super(DNN, self).__init__()

        self.layers = nn.ModuleList()

        for i in range(0, len(layer_sizes) - 1):
            self.layers.append(Linear(in_features=layer_sizes[i],
                                      out_features=layer_sizes[i + 1]))
            self.layers.append(ReLU())

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            # print(f'Layer {layer} with input size {x.size()}')
            x = layer(x)
        return x


class SparseDataset(Dataset):
    """
    Custom Dataset class for scipy sparse matrix
    """

    def __init__(self, data: Union[np.ndarray, coo_matrix, csr_matrix],
                 targets: Union[np.ndarray, coo_matrix, csr_matrix]):
        # Transform data coo_matrix to csr_matrix for indexing
        if type(data) == coo_matrix:
            self.data = data.tocsr()
        else:
            self.data = data

        # Transform targets coo_matrix to csr_matrix for indexing
        if type(targets) == coo_matrix:
            self.targets = targets.tocsr()
        else:
            self.targets = targets

    def __getitem__(self, index: int):
        return self.data[index], self.targets[index]

    def __len__(self):
        return self.data.shape[0]


dataloader = DataLoader(
    SparseDataset(data=X_train, targets=Y_train),
    batch_size=128,
    shuffle=True
)

for i, batch in enumerate(dataloader):
    print(f'i: {batch}')
