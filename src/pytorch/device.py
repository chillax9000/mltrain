import torch

from command import CmdArg


def get_device_label_from_args(args):
    device_label = "cpu"
    if args[CmdArg.cuda]:
        if not torch.cuda.is_available():
            print("CUDA not available on this machine")
            exit(0)
        device_label = "cuda"
    return device_label


def get_device_from_args(args):
    return torch.device(get_device_label_from_args(args))