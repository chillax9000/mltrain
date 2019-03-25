import torch
import torch.optim as optim
import torch.nn as nn


def make_print_fn(print_every, n_iter):
    if print_every < 0:
        return lambda *x, **y: None
    else:
        def fn(step, loss):
            if step % print_every == 0:
                print(f"{step}/{n_iter}: {loss}")
        return fn


def get_random_input_target(data):
    return torch.tensor([1, 2]), torch.tensor([3])


def train(model, input, target, criterion, optimizer):
    optimizer.zero_grad()
    output = model(input)
    loss = criterion(output.unsqueeze(dim=0), target)
    loss.backward()
    optimizer.step()
    return loss


def do_training(model, data, n_iter, criterion=None, optimizer=None, print_every=-1):
    if criterion is None:
        criterion = nn.NLLLoss()
    if optimizer is None:
        optimizer = optim.Adam(model.parameters())

    print_fn = make_print_fn(print_every, n_iter)

    for step in range(1, n_iter + 1):
        input, target = get_random_input_target(data)
        loss = train(model, input, target, criterion, optimizer)

        print_fn(step, loss)
