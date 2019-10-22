import torch

import torch.nn as nn

default_device = torch.device("cpu")


def get_nn(n_input, n_hidden, n_output):
    return nn.Sequential(
        nn.Linear(n_input, n_hidden),
        nn.ReLU(),
        nn.Linear(n_hidden, n_output),
        nn.LogSoftmax(dim=0)
    )


class WordEmbSkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, hidden_layer_size, device=default_device):
        super().__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim).to(device=device)
        self.net = nn.Sequential(
            nn.Linear(context_size * embedding_dim, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, vocab_size),
            nn.LogSoftmax(dim=1)).to(device=device)

    def forward(self, indexes):
        output = self.embedding(indexes).reshape(indexes.shape[0], 1, -1).squeeze(1)  # can't directly reshape('shape_0', -1) ?
        output = self.net(output)
        return output
