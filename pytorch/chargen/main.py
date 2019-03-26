# from https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html

from __future__ import unicode_literals, print_function, division

import os

import command
import pytorch.chargen.modelbuilder as modelbuilder
from command import CmdArg
from pytorch import serialize
from pytorch.chargen.generate import samples_nn_rnn, samples
from pytorch.chargen.train import do_training

serializer = serialize.ModelSerializer(os.path.dirname(__file__))

mode, args = command.create_parser_and_parse()

if mode == "list":
    modelbuilder.print_models_list()
    exit(0)

if mode == "list-dumps":
    print("Available dumps")
    for dump in serializer.get_dumps():
        print("+", dump)
    exit(0)

if mode == "train":
    if not args[CmdArg.model]:
        print(f"A model is required, specify it with option {CmdArg.model.cmd_name}")
        modelbuilder.print_models_list()
        exit(0)
    try:
        builder = modelbuilder.get_builder(args[CmdArg.model])
    except ValueError:
        print(f"Could not find a model named: {args[CmdArg.model]}")
        modelbuilder.print_models_list()
        exit(0)
    model, data, train_fun = builder.build(args)

    print(f"Running in mode {mode}, with args:")
    for arg, val in args.items():
        print(arg, ":", val)
    print()

    do_training(model, data, train_fun, n_iter=args[CmdArg.iter])
    serializer.dump(model, args)

if mode == "test":
    model_folder_name = args.get(CmdArg.dump_name, None)
    if model_folder_name is None:
        print("Must specify a model folder to load with option --dump-name")
        exit(0)
    model, data, _ = serializer.load(model_folder_name, args)
    for category in data.all_categories:
        print()
        print(category)
        samples(model, data, category, data.all_chars)
        # samples_nn_rnn(model, data, category, data.all_chars)
