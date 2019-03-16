# from https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html

from __future__ import unicode_literals, print_function, division

import os

from pytorch.chargen.generate import samples
from pytorch.chargen.model import RNN
from pytorch.chargen.train import do_training
from pytorch.chargen.init import n_letters, project_path, all_letters, all_categories, n_categories
import torch

# True to recompute ##########
#
train_model = True

n_iter = 100_000
device_label = "cpu"  # cuda or cpu
device = torch.device(device_label)
model = RNN(n_letters, 1024, n_letters, n_categories, device)
#
##############################

path_model_save = os.path.join(project_path, "saved", "model.pt")
if train_model:
    do_training(model, n_iter=n_iter)
    torch.save(model.state_dict(), path_model_save)
else:
    model.load_state_dict(torch.load(path_model_save))

for category in all_categories:
    print(category)
    samples(model, category, all_letters)
