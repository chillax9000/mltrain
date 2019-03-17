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
        self.softmax = nn.LogSoftmax(dim=1).to(device=device)

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size).to(device=self.device)
