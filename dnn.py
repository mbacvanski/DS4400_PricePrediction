import faulthandler
import pickle
from typing import List, Tuple

import numpy as np
import torch
import tqdm
from scipy.sparse import csr_matrix
from torch import nn, optim
from torch.nn import Linear, ReLU

faulthandler.enable()


class DNN(nn.Module):
    def __init__(self, layer_sizes: List[int]):
        """
        The first layer size is the input dimension, and the last layer size is the output dimension.
        """
        super(DNN, self).__init__()

        self.layers = nn.ModuleList()

        for i in range(0, len(layer_sizes) - 1):
            self.layers.append(Linear(in_features=layer_sizes[i],
                                      out_features=layer_sizes[i + 1]))
            self.layers.append(ReLU())

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            # print(f'Layer {layer} with input size {x.size()} of type {x.dtype}')
            x = layer(x)
        return x


def train_model(model, train: Tuple[csr_matrix, np.ndarray], validation: Tuple[csr_matrix, np.ndarray],
                batch_size: int, epochs: int):
    optimizer = optim.Adam(model.parameters())

    X_train, Y_train = train
    n_samples = X_train.shape[0]
    n_batches = int(np.ceil(n_samples * 1.0 / batch_size))
    display = tqdm.trange(epochs, disable=True)
    validation_losses = []
    for _ in display:
        permutation = np.random.permutation(n_samples)

        # train in this epoch
        inner_loop = tqdm.tqdm(range(n_batches))
        for batch_n in inner_loop:
            from_idx = batch_n * batch_size
            to_idx = min((batch_n + 1) * batch_size, len(permutation))
            n_in_this_batch = to_idx - from_idx
            batch_indices = permutation[from_idx: to_idx]
            batch_x = torch.as_tensor(np.float32(X_train[batch_indices].todense())[:, :INPUT_SIZE],
                                      # batch_x = torch.as_tensor(np.float32(X_train[batch_indices].todense()),
                                      dtype=torch.float32)

            # if this doesn't work, try the line commented out above
            # batch_x = csr_to_tensor(X_train[batch_indices])
            batch_y = torch.as_tensor(np.float32(Y_train[batch_indices]).reshape(-1, 1),
                                      dtype=torch.float32)
            # batch_y.resize(n_in_this_batch, 1)

            predictions = model(batch_x)
            loss = nn.MSELoss()(predictions, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # inner_loop.set_description(f'')
            inner_loop.set_description_str(f'Loss: {loss.item()}')

    return validation_losses


def csr_to_tensor(x: csr_matrix) -> torch.Tensor:
    coo = x.tocoo()
    return torch.sparse.FloatTensor(torch.LongTensor([coo.row.tolist(), coo.col.tolist()]),
                                    torch.FloatTensor(coo.data))


INPUT_SIZE = 1_000


def main():
    with open('x_train_transformed.pkl', 'rb') as f:
        X_train = pickle.load(f)

    with open('x_test_transformed.pkl', 'rb') as f:
        X_test = pickle.load(f)

    with open('y_train.pkl', 'rb') as f:
        Y_train = pickle.load(f)

    with open('y_test.pkl', 'rb') as f:
        Y_test = pickle.load(f)

    dnn = DNN(layer_sizes=[INPUT_SIZE, 1])
    # dnn = DNN(layer_sizes=[X_train.shape[1], 256, 1])

    train_model(model=dnn, train=(X_train, Y_train), validation=(X_test, Y_test), batch_size=25, epochs=1)

    torch.save(dnn.state_dict(), f'dnn_model_{INPUT_SIZE}_1.pt')


if __name__ == '__main__':
    main()
