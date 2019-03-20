# from https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html

from __future__ import unicode_literals, print_function, division

import json
import os

import torch

import command
from command import CmdArg
from pytorch.chargen.data import project_path
from pytorch.chargen.generate import samples_nn_rnn, samples
from pytorch.chargen.train import do_training
import pytorch.chargen.modelbuilder as modelbuilder
from pytorch.device import get_device_label_from_args


def save_model(model, args):
    torch.save(model.state_dict(), path_model_save)
    with open(os.path.join(project_path, "saved", "model_info.json"), "w") as fp:
        json.dump({arg.name: val for arg, val in args.items()}, fp)


def load_args():
    with open(os.path.join(project_path, "saved", "model_info.json")) as fp:
        str_dict = json.load(fp)
    return {CmdArg.decode(str_arg): val for str_arg, val in str_dict.items()}


def print_models_list():
    print("Available models:")
    for model_name in modelbuilder.MODELS:
        print("+", model_name)


path_model_save = os.path.join(project_path, "saved", "model.pt")
mode, args = command.create_parser_and_parse()

if mode == "list":
    print_models_list()
    exit(0)

if mode == "test":  # reusing saved options, keeping actual command options (like cuda)
    saved_args = load_args()
    saved_args.update(args)
    args = saved_args

if not args[CmdArg.model]:
    print(f"A model is required, specify it with option {CmdArg.model.cmd_name}")
    print_models_list()
    exit(0)
builder = modelbuilder.get_model(args[CmdArg.model])
if builder is None:
    print(f"Could not find a model named: {args[CmdArg.model]}")
    print_models_list()
    exit(0)
model, data, train_fun = builder.build(args)


print(f"Running in mode {mode}, with args:")
for arg, val in args.items():
    print(arg, ":", val)
print()

if mode == "train":
    do_training(model, data, train_fun, n_iter=args[CmdArg.iter])
    save_model(model, args)

elif mode == "test":
    model.load_state_dict(torch.load(path_model_save, map_location=get_device_label_from_args(args)))
    model.eval()
    for category in data.all_categories:
        print()
        print(category)
        # samples(model, data, category, data.all_chars)
        samples_nn_rnn(model, data, category, data.all_chars)
