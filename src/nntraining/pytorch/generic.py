import os

import matplotlib.pyplot as plt
import torch
import torch.utils.data

import nntraining.clock


def train(model, input, target, criterion, optimizer):
    optimizer.zero_grad()
    output = model(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    return loss


def do_training(model, dataset, fun_train, model_folder_path, criterion=None, optimizer=None,
                n_iter=10000, print_every=500, plot_every=500):
    if optimizer is None:
        optimizer = torch.optim.Adagrad(model.parameters())
    if criterion is None:
        criterion = torch.nn.NLLLoss().to(device=model.device)

    all_losses = []
    total_loss = 0

    watch = nntraining.clock.Clock()
    watch.start()

    loader = LoaderWrapper(torch.utils.data.DataLoader(dataset), n_iter=n_iter)
    for step, (input, target) in enumerate(loader):
        step += 1
        try:
            loss = fun_train(model, input, target, criterion=criterion, optimizer=optimizer)
            total_loss += loss

            if print_every > 0 and step % print_every == 0:
                print(f"{100 * step / n_iter:>5.1f}%: {loss:6.2f} "
                      f"({watch.call_and_get_elapsed_since_last_call():.2f}s, "
                      f"total {watch.get_elapsed_since_start():.2f}s)")

            if step % plot_every == 0:
                all_losses.append(total_loss / plot_every)
                total_loss = 0
        except Exception as e:
            print("a training iteration went wrong:")
            print(e)
            print("input:", input)
            print("target:", target)

    plt.figure(figsize=(20, 10))
    plt.plot(all_losses)
    if not os.path.exists(model_folder_path):
        os.makedirs(model_folder_path)
    plt.savefig(os.path.join(model_folder_path, "train_loss.png"))
    return model


class LoaderWrapper:
    def __init__(self, dataloader, n_iter):
        self.n_iter = n_iter
        self.dataloader = dataloader
        self.iterator = iter(dataloader)
        self.count = 0

    def __iter__(self):
        return self

    def __next__(self):

        if self.count >= self.n_iter:
            raise StopIteration

        try:
            to_return = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            to_return = next(self.iterator)

        self.count += 1
        return to_return
