# from https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html

from __future__ import unicode_literals, print_function, division

import os

import torch

from pytorch.chargen.generate import samples
from pytorch.chargen.data import project_path, Data
from pytorch.chargen.model import RNN
from pytorch.chargen.train import do_training

import argparse

# Default options ############
#
train_model = True

device_label = "cpu"
#
##############################

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true', help='Use CUDA')
parser.add_argument('--iter', help='Number of training iterations', type=int, default=1_000_000)
parser.add_argument('--hidden', help='Number of nodes in the hidden layer', type=int, default=1024)

args = parser.parse_args()
if args.cuda:
    if not torch.cuda.is_available():
        print("CUDA not available on this machine")
        exit(0)
    device_label = "cuda"

device = torch.device(device_label)
n_iter = args.iter
n_hidden_nodes = args.hidden

data = Data()
model = RNN(data.n_letters, n_hidden_nodes, data.n_letters, data.n_categories, device)

path_model_save = os.path.join(project_path, "saved", "model.pt")
if train_model:
    do_training(model, data, n_iter=n_iter)
    torch.save(model.state_dict(), path_model_save)
else:
    model.load_state_dict(torch.load(path_model_save))

for category in data.all_categories:
    print(category)
    samples(model, data, category, data.all_letters)
