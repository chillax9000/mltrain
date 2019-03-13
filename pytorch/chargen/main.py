# from https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html

from __future__ import unicode_literals, print_function, division

import os

from pytorch.chargen.generate import samples
from pytorch.chargen.model import RNN
from pytorch.chargen.train import do_training
from pytorch.chargen.util import n_letters, project_path, all_letters
import torch

# True to recompute #
                    #
train_model = True  #
                    #
#####################

path_model_save = os.path.join(project_path, "saved", "model.pt")
rnn = RNN(n_letters, 128, n_letters)
if train_model:
    do_training(rnn, n_iters=10000)
    torch.save(rnn.state_dict(), path_model_save)
else:
    rnn.load_state_dict(torch.load(path_model_save))

print(samples(rnn, "fr", all_letters))
