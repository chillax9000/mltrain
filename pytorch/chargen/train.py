import os

import torch
import torch.nn as nn
import random
from pytorch.chargen.init import all_letters, all_categories, category_lines, n_categories, n_letters, project_path
import clock
import matplotlib.pyplot as plt

default_device = torch.device("cpu")


# Random item from a list
def random_choice(l):
    return l[random.randint(0, len(l) - 1)]


# Get a random category and random line from that category
def get_random_training_pair():
    category = random_choice(all_categories)
    line = random_choice(category_lines[category])
    return category, line


# One-hot vector for category
def get_category_tensor(category, device=default_device):
    li = all_categories.index(category)
    tensor = torch.zeros(1, n_categories).to(device=device)
    tensor[0][li] = 1
    return tensor


# One-hot matrix of first to last letters (not including EOS) for input
def get_input_tensor(line, device=default_device):
    tensor = torch.zeros(len(line), 1, n_letters).to(device=device)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor


# LongTensor of second letter to end (EOS) for target
def get_target_tensor(line, device=default_device):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1)  # EOS
    return torch.LongTensor(letter_indexes).to(device=device)


# Make category, input, and target tensors from a random category, line pair
def random_training_example(device=default_device):
    category, line = get_random_training_pair()
    category_tensor = get_category_tensor(category, device)
    input_line_tensor = get_input_tensor(line, device)
    target_line_tensor = get_target_tensor(line, device)
    return category_tensor, input_line_tensor, target_line_tensor


def train(rnn, category_tensor, input_line_tensor, target_line_tensor, criterion, learning_rate=0.0005):
    target_line_tensor.unsqueeze_(-1)
    hidden = rnn.initHidden()

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


def do_training(rnn, criterion=None, n_iter=10000, print_every=500, plot_every=500):
    if not criterion:
        criterion = nn.NLLLoss().to(device=rnn.device)
    all_losses = []
    total_loss = 0

    watch = clock.Clock()
    watch.start()

    for iter in range(1, n_iter + 1):
        try:
            random_example = random_training_example(rnn.device)
            output, loss = train(rnn, *random_example, criterion=criterion)
            total_loss += loss

            if iter % print_every == 0:
                print('%.2fs (%d %d%%) %.4f' % (watch.elapsed_since_start(), iter, iter / n_iter * 100, loss))

            if iter % plot_every == 0:
                all_losses.append(total_loss / plot_every)
                total_loss = 0
        except Exception as e:
            print("an training iteration went wrong:")
            print(e)
            print(random_example)

    plt.figure(figsize=(20, 10))
    plt.plot(all_losses)
    plt.savefig(os.path.join(project_path, "saved", "train_loss.png"))
    return rnn
