import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data


def make_print_fn(print_every, n_iter):
    if print_every < 0:
        return lambda *x, **y: None
    else:
        def fn(step, loss):
            if step % print_every == 0:
                print(f"{step}/{n_iter}: {loss}")
        return fn


def get_random_input_target(data):
    indexes = data.get_trigram_indexes()
    input_indexes = indexes[0: 2]
    target_index = [indexes[2]]
    return torch.tensor(input_indexes), torch.tensor(target_index)


def train(model, input, target, criterion, optimizer):
    optimizer.zero_grad()
    output = model(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    return loss


def do_training(model, dataset, train_fun, n_iter, criterion=None, optimizer=None, print_every=-1):
    if criterion is None:
        criterion = nn.NLLLoss()
    if optimizer is None:
        optimizer = optim.Adagrad(model.parameters())

    print_fn = make_print_fn(print_every, n_iter)

    for step, (input, target) in enumerate(torch.utils.data.DataLoader(dataset)):
        loss = train_fun(model, input, target, criterion, optimizer)

        print_fn(step, loss)
