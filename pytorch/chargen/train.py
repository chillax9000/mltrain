import os

import torch
import torch.nn as nn
import random
from pytorch.chargen.init import all_letters, all_categories, category_lines, n_categories, n_letters, project_path
import clock
import matplotlib.pyplot as plt


# Random item from a list
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


# Get a random category and random line from that category
def randomTrainingPair():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    return category, line


# One-hot vector for category
def categoryTensor(category):
    li = all_categories.index(category)
    tensor = torch.zeros(1, n_categories)
    tensor[0][li] = 1
    return tensor


# One-hot matrix of first to last letters (not including EOS) for input
def inputTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor


# LongTensor of second letter to end (EOS) for target
def targetTensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1)  # EOS
    return torch.LongTensor(letter_indexes)


# Make category, input, and target tensors from a random category, line pair
def randomTrainingExample():
    category, line = randomTrainingPair()
    category_tensor = categoryTensor(category)
    input_line_tensor = inputTensor(line)
    target_line_tensor = targetTensor(line)
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
        criterion = nn.NLLLoss()
    all_losses = []
    total_loss = 0

    watch = clock.Clock()
    watch.start()

    for iter in range(1, n_iter + 1):
        try:
            random_example = randomTrainingExample()
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

    plt.figure()
    plt.plot(all_losses)
    plt.savefig(os.path.join(project_path, "saved", "train_loss.png"))
    return rnn
