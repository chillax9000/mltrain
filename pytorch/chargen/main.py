# from https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html

from __future__ import unicode_literals, print_function, division

from pytorch.chargen.generate import sample
from pytorch.chargen.model import RNN
from pytorch.chargen.train import do_training
from pytorch.chargen.util import n_letters

rnn = RNN(n_letters, 32, n_letters)

do_training(rnn, n_iters=5000)

print(sample(rnn, "blab", "a"))
