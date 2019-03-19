# from https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html

from __future__ import unicode_literals, print_function, division

import json
import os

import torch

from pytorch.chargen import command, train
from pytorch.chargen.command import CmdArg
from pytorch.chargen.data import project_path, Data, DataWord, DataSentence
from pytorch.chargen.generate import samples_nn_rnn, samples
from pytorch.chargen.model import SimpleRNN, RNN
from pytorch.chargen.train import do_training


def save_model(model, args):
    torch.save(model.state_dict(), path_model_save)
    with open(os.path.join(project_path, "saved", "model_info.json"), "w") as fp:
        json.dump({arg.name: val for arg, val in args.items()}, fp)


def load_args():
    with open(os.path.join(project_path, "saved", "model_info.json")) as fp:
        str_dict = json.load(fp)
    return {CmdArg.decode(str_arg): val for str_arg, val in str_dict.items()}


path_model_save = os.path.join(project_path, "saved", "model.pt")
mode, args = command.create_parser_and_parse()

if mode == "test":  # reusing saved options, keeping actual command options (like cuda)
    saved_args = load_args()
    saved_args.update(args)
    args = saved_args

print(f"Running in mode {mode}, with args:")
for arg, val in args.items():
    print(arg, ":", val)
print()

device_label = "cpu"
if args[CmdArg.cuda]:
    if not torch.cuda.is_available():
        print("CUDA not available on this machine")
        exit(0)
    device_label = "cuda"
device = torch.device(device_label)
n_iter = args[CmdArg.iter]
size_hidden = args[CmdArg.hidden]

data = DataSentence(device)
# model = RNN(data.n_chars, size_hidden, data.n_chars, data.n_categories, device)
# fun_train = train.train

model = SimpleRNN(data.n_chars, data.n_categories, size_hidden, device=device)
fun_train = train.train_nn_rnn

if mode == "train":
    do_training(model, data, fun_train, n_iter=n_iter)
    save_model(model, args)

elif mode == "test":
    model.load_state_dict(torch.load(path_model_save, map_location=device_label))
    model.eval()
    for category in data.all_categories:
        print()
        print(category)
        # samples(model, data, category, data.all_chars)
        samples_nn_rnn(model, data, category, data.all_chars)
