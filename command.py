import argparse
import enum
from typing import List


class CmdArg(enum.Enum):
    cuda = ("--cuda", {"action": 'store_true',
                       "help": 'Use CUDA'})
    iter = ("--iter", {"help": 'Number of training iterations',
                       "type": int,
                       "default": 1_000_000})
    hidden = ("--hidden", {"help": 'Size of the hidden layer',
                           "type": int,
                           "default": 1024})
    embedding = ("--embedding", {"help": 'Dimension of the embedding',
                                 "type": int,
                                 "default": 256})
    context = ("--context", {"help": 'Size of the context window',
                             "type": int,
                             "default": 2})
    model = ("--model", {"help": "Model to train",
                         "type": str,
                         "default": None})
    dump_name = ("--dump-name", {"help": "name of the folder to load",
                                 "type": str,
                                 "default": None})

    def __init__(self, cmd_name, settings):
        self.cmd_name = cmd_name
        self.settings = settings

    def __repr__(self):
        return f"{type(self).__name__}.{self.name}"

    @classmethod
    def decode(cls, s: str):
        for arg in cls:
            if s == arg.name:
                return arg


class CommandConfig:
    def __init__(self, train=None, test=None):
        self.train_args: List[CmdArg] = [] if not train else train
        self.test_args: List[CmdArg] = [] if not test else test

    @classmethod
    def default(cls):
        return cls(train=list(CmdArg), test=[CmdArg.cuda, CmdArg.dump_name])


def create_parser(command_config: CommandConfig = CommandConfig.default()) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_train = subparsers.add_parser("train")
    for arg in command_config.train_args:
        parser_train.add_argument(arg.cmd_name, **arg.settings)
    parser_train.set_defaults(subparser="train")

    parser_test = subparsers.add_parser("test")
    for arg in command_config.test_args:
        parser_test.add_argument(arg.cmd_name, **arg.settings)
    parser_test.set_defaults(subparser="test")

    parser_list_models = subparsers.add_parser("list")
    parser_list_models.set_defaults(subparser="list")

    parser_list_models = subparsers.add_parser("list-dumps")
    parser_list_models.set_defaults(subparser="list-dumps")
    return parser


def create_parser_and_parse(command_config: CommandConfig = CommandConfig.default()):
    parser = create_parser(command_config)
    dict_args_raw = vars(parser.parse_args())
    subparser_chosen = dict_args_raw["subparser"]
    del dict_args_raw["subparser"]

    dict_args = {arg: dict_args_raw[arg.name] for arg in CmdArg if arg.name in dict_args_raw}

    return subparser_chosen, dict_args
