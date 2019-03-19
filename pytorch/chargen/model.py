import torch
import torch.nn as nn

default_device = torch.device("cpu")


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_categories, device=default_device):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.device = device

        self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size).to(device=device)
        self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size).to(device=device)
        self.o2o = nn.Linear(hidden_size + output_size, output_size).to(device=device)
        self.dropout = nn.Dropout(0.1).to(device=device)
        self.softmax = nn.LogSoftmax(dim=0).to(device=device)

    def forward(self, category, line, hidden):
        input_combined = torch.cat((category, line, hidden))
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output))
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(self.hidden_size).to(device=self.device)


class SimpleRNN(nn.Module):
    def __init__(self, n_letters, n_categories, size_hidden, num_layers=2, dropout=0.1, device=default_device):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size=n_categories + n_letters,
                 hidden_size=size_hidden,
                 num_layers=num_layers,
                 dropout=dropout
                 ).to(device=device)
        self.last = nn.Linear(size_hidden, n_letters).to(device=device)
        self.device = device
        self.size_hidden = size_hidden

    def forward(self, input, hidden=None):
        output, hidden = self.rnn(input) if hidden is None else self.rnn(input, hidden)
        output = self.last(output)
        return output, hidden
