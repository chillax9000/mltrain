from __future__ import unicode_literals, print_function, division

import os

import command
from pytorch import modelbuilder
from command import CmdArg
from pytorch import serialize
from pytorch import generic

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
    try:
        model, dataset, train_fun = modelbuilder.build_from_args(args)
    except ValueError as e:
        print(e)
        modelbuilder.print_models_list()
        exit(0)

    print(f"Running in mode {mode}, with args:")
    for arg, val in args.items():
        print(arg, ":", val)
    print()

    replace_last = False
    folder_path, _, _ = serializer.get_dump_paths(replace_last)
    generic.do_training(model, dataset, train_fun, model_folder_path=folder_path, n_iter=args[CmdArg.iter])
    serializer.dump(model, args, replace_last)

if mode == "test":
    print("no test here")
