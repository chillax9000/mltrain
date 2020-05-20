from __future__ import unicode_literals, print_function, division

import os

import mlt.command
from mlt.pytorch import modelbuilder
from mlt.command import CmdArg
from mlt.pytorch import serialize
from mlt.pytorch import generic


def main():
    mode, args = mlt.command.create_parser_and_parse()
    replace = args[CmdArg.replace] if mode == "train" else False
    serializer = serialize.ModelSerializer(os.path.dirname(__file__), replace=replace)

    if mode == "list":
        modelbuilder.print_model_list()
        exit(0)

    if mode == "list-dumps":
        serializer.print_dump_list()
        exit(0)

    if mode == "train":
        try:
            model, dataset, train_fun = modelbuilder.build_from_args(args)
        except ValueError as e:
            print(e)
            modelbuilder.print_model_list()
            exit(0)

        print(f"Running in mode {mode}, with args:")
        for arg, val in args.items():
            print(arg, ":", val)
        print()

        generic.do_training(model, dataset, train_fun, model_folder_path=serializer.dump_folder_path,
                            n_iter=args[CmdArg.iter])
        serializer.dump(model, args)

    if mode == "test":
        print("no test here")
