import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim

import clock
from pytorch.chargen.data import project_path


# LongTensor of second char to end (EOS) for target
def get_target_tensor(data, line):
    char_indexes = [data.get_char_index(l) for l in line[1:]]
    char_indexes.append(data.n_chars - 1)  # EOS
    return torch.LongTensor(char_indexes).to(device=data.device)


def random_training_example(data):
    """returns a tuple (category_tensor, line_tensor, target_tensor)"""
    category, line = data.get_random_training_pair()
    return (data.get_category_tensor(category),
            data.get_line_tensor(line),
            get_target_tensor(data, line))


def get_input_from_category_and_line_tensors(category_tensor, line_tensor):
    """ returns (N, 1, A+B) sized tensor from (A, ) and (N, B) sized tensors"""
    category_tensor = category_tensor.unsqueeze(0).expand(line_tensor.size(0), -1)
    return torch.cat((category_tensor, line_tensor), 1).unsqueeze(1)


def train(rnn, category_tensor, line_tensor, target_tensor, criterion, optimizer):
    hidden = rnn.init_hidden()

    optimizer.zero_grad()

    loss = 0

    for i in range(line_tensor.size(0)):
        output, hidden = rnn(category_tensor, line_tensor[i], hidden)
        l = criterion(output.unsqueeze(0), target_tensor[i].unsqueeze(-1))
        loss += l

    loss.backward()

    optimizer.step()

    return output, loss.item() / line_tensor.size(0)


def train_nn_rnn(rnn, category_tensor, line_tensor, target_tensor, criterion, optimizer):
    optimizer.zero_grad()

    input = get_input_from_category_and_line_tensors(category_tensor, line_tensor)
    output, _ = rnn(input)

    m = nn.LogSoftmax(dim=1)
    loss = criterion(m(output.squeeze(1)), target_tensor)
    loss.backward()

    optimizer.step()

    return output, loss.item() / line_tensor.size(0)


def do_training(rnn, data, fun_train, criterion=None, optimizer=None, n_iter=10000, print_every=500, plot_every=500):
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
            random_example = random_training_example(data)
            _, loss = fun_train(rnn, *random_example, criterion=criterion, optimizer=optimizer)
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
            raise e

    plt.figure(figsize=(20, 10))
    plt.plot(all_losses)
    plt.savefig(os.path.join(project_path, "saved", "train_loss.png"))
    return rnn
