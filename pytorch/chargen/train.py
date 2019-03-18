import os
import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import clock
import util.vector as vector
from pytorch.chargen.data import project_path
import torch.optim

default_device = torch.device("cpu")


# Random item from a list
def random_choice(l):
    return l[random.randrange(0, len(l))]


# Get a random category and random line from that category
def get_random_training_pair(data):
    category = random_choice(data.all_categories)
    line = random_choice(data.category_lines[category])
    return category, line


# Store tensors for reuse
def create_tensors(data, device=default_device):
    tensors = {
        "category": {},
        "letter": {}
    }
    for idx, category in enumerate(data.all_categories):
        tensors["category"][category] = torch.Tensor(vector.one_hot(data.n_categories, idx)).to(device=device)
    for idx, letter in enumerate(data.all_letters):
        tensors["letter"][letter] = torch.Tensor(vector.one_hot(data.n_letters, idx)).to(device=device)
    return tensors


# One-hot vector for category
def get_category_tensor(tensors, category):
    return tensors["category"][category]


# One-hot matrix of first to last letters (not including EOS) for input
def get_input_tensor(tensors, line):
    return torch.cat(tuple(tensors["letter"][letter] for letter in line)).unsqueeze(1)


# LongTensor of second letter to end (EOS) for target
def get_target_tensor(data, line, device=default_device):
    letter_indexes = [data.all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(data.n_letters - 1)  # EOS
    return torch.LongTensor(letter_indexes).to(device=device)


# Make category, input, and target tensors from a random category, line pair
def random_training_example(tensors, data, device=default_device):
    category, line = get_random_training_pair(data)
    category_tensor = get_category_tensor(tensors, category)
    input_line_tensor = get_input_tensor(tensors, line)
    target_line_tensor = get_target_tensor(data, line, device)
    return category_tensor, input_line_tensor, target_line_tensor


def train(rnn, category_tensor, input_line_tensor, target_line_tensor, criterion, learning_rate=0.0005):
    target_line_tensor.unsqueeze_(-1)
    hidden = rnn.init_hidden()

    rnn.zero_grad()

    loss = 0

    for i in range(input_line_tensor.size(0)):
        output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
        l = criterion(output, target_line_tensor[i])
        loss += l

    loss.backward()

    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)
    return output, loss.item() / input_line_tensor.size(0)


def train_nn_rnn(rnn, category_tensor, input_line_tensor, target_line_tensor, criterion, optimizer):
    target_line_tensor.unsqueeze_(-1)

    optimizer.zero_grad()

    m = nn.LogSoftmax(dim=1)
    category_tensor = category_tensor.unsqueeze(1).expand(input_line_tensor.size(0), -1, -1)
    output, hidden = rnn(torch.cat((category_tensor, input_line_tensor), 2))
    loss = criterion(m(output.squeeze(1)), target_line_tensor.squeeze(1))

    loss.backward()

    optimizer.step()
    return output, loss.item() / input_line_tensor.size(0)


def do_training(rnn, data, criterion=None, optimizer=None, n_iter=10000, print_every=500, plot_every=500):
    tensors = create_tensors(data, rnn.device)
    if optimizer is None:
        optimizer = torch.optim.Adagrad(rnn.parameters())
    if criterion is None:
        criterion = nn.NLLLoss().to(device=rnn.device)

    all_losses = []
    total_loss = 0

    watch = clock.Clock()
    watch.start()

    for iter in range(1, n_iter + 1):
        try:
            random_example = random_training_example(tensors, data, rnn.device)
            _, loss = train_nn_rnn(rnn, *random_example, criterion=criterion, optimizer=optimizer)
            total_loss += loss

            if iter % print_every == 0:
                print('%.2fs (%d %d%%) %.4f' % (watch.elapsed_since_start(), iter, iter / n_iter * 100, loss))

            if iter % plot_every == 0:
                all_losses.append(total_loss / plot_every)
                total_loss = 0
        except Exception as e:
            print("a training iteration went wrong:")
            print(e)
            print(random_example)

    plt.figure(figsize=(20, 10))
    plt.plot(all_losses)
    plt.savefig(os.path.join(project_path, "saved", "train_loss.png"))
    return rnn