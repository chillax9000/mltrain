import os

import matplotlib.pyplot as plt
import torch

import clock


def do_training(rnn, dataloader, fun_train, model_folder_path, criterion=None, optimizer=None,
                n_iter=10000, print_every=500, plot_every=500):
    if optimizer is None:
        optimizer = torch.optim.Adagrad(rnn.parameters())
    if criterion is None:
        criterion = torch.nn.NLLLoss().to(device=rnn.device)

    all_losses = []
    total_loss = 0

    watch = clock.Clock()
    watch.start()

    for iter in range(1, n_iter + 1):
        try:
            input, target = dataloader()
            loss = fun_train(rnn, input, target, criterion=criterion, optimizer=optimizer)
            total_loss += loss

            if iter % print_every == 0:
                print('%.2fs (%d %d%%) %.4f' % (watch.elapsed_since_start(), iter, iter / n_iter * 100, loss))

            if iter % plot_every == 0:
                all_losses.append(total_loss / plot_every)
                total_loss = 0
        except Exception as e:
            print("a training iteration went wrong:")
            print(e)
            print("input:", input)
            print("target:", target)

    plt.figure(figsize=(20, 10))
    plt.plot(all_losses)
    plt.savefig(os.path.join(model_folder_path, "train_loss.png"))
    return rnn
