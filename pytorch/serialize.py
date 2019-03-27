import collections
import json
import os
import re
import shutil
from operator import attrgetter

import torch

from command import CmdArg
from pytorch import modelbuilder
from pytorch.device import get_device_label_from_args


def build_model_info_dict(model):
    return {
        "model_name": model._serialize_name
    }


Dump = collections.namedtuple("Dump", "name info")


class ModelSerializer:
    """ models are saved in a subfolder of self.project_saved_path which contains the serialization + a json for
    model info and creation args (from cli)"""

    def __init__(self, project_path, save_dir="saved", model_folder_prefix="model_"):
        self.project_path = project_path
        self.project_save_path = os.path.join(self.project_path, save_dir)
        self.model_filename = "model"
        self.json_filename = "info.json"
        self.model_folder_prefix = model_folder_prefix
        self.model_folder_name_regex = re.compile(self.model_folder_prefix + "[0-9]+")

        if not os.path.exists(self.project_save_path):
            os.makedirs(self.project_save_path)

    def is_model_folder_like_name(self, s: str):
        return re.fullmatch(self.model_folder_name_regex, s) is not None

    def get_model_folder_number(self, model_folder_name):
        return int(model_folder_name[len(self.model_folder_prefix):])

    def get_dumps(self):
        dumps_names = filter(self.is_model_folder_like_name,
                             map(attrgetter("name"),
                                 filter(os.DirEntry.is_dir,
                                        os.scandir(self.project_save_path))))

        def get_info(dump_name):
            try:
                _, info = self.load_json(dump_name)
                return info
            except:
                return "something is wrong"

        return [Dump(name, get_info(name)) for name in dumps_names]

    def get_latest_dump_number(self):
        numbers = set(map(self.get_model_folder_number,
                          map(attrgetter("name"),
                              self.get_dumps())))
        return max(numbers) if numbers else 0

    def get_dump_paths(self, replace_last=False):
        """returns: folder_path, model_path, json_path"""
        number = self.get_latest_dump_number() + (0 if replace_last else 1)
        model_folder_name = f"{self.model_folder_prefix}{number}"
        folder_path = os.path.join(self.project_save_path, model_folder_name)
        model_path = os.path.join(folder_path, self.model_filename)
        json_path = os.path.join(folder_path, self.json_filename)
        return folder_path, model_path, json_path

    def dump(self, model, args, replace_last=False):
        folder_path, model_path, json_path = self.get_dump_paths(replace_last)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        os.makedirs(folder_path)
        torch.save(model.state_dict(), model_path)
        with open(json_path, "w") as fp:
            final_dict = {
                "info": build_model_info_dict(model),
                "args": {arg.name: val for arg, val in args.items()},
            }
            json.dump(final_dict, fp)
        print(f"saved model into {folder_path}")

    def get_json_load_path(self, model_folder_name):
        return os.path.join(self.project_save_path, model_folder_name, self.json_filename)

    def get_model_load_path(self, model_folder_name):
        return os.path.join(self.project_save_path, model_folder_name, self.model_filename)

    def load_json(self, model_folder_name):
        """returns tuple of dicts: args (with CmdArg keys), info"""
        with open(self.get_json_load_path(model_folder_name)) as fp:
            str_dict = json.load(fp)
        args = str_dict["args"]
        info = str_dict["info"]
        return {CmdArg.decode(str_arg): val for str_arg, val in args.items()}, info

    def get_model_name(self, model_folder_name):
        with open(self.get_json_load_path(model_folder_name)) as fp:
            str_dict = json.load(fp)
        return str_dict["info"]["model_name"]

    def load(self, model_folder_name, new_args):
        """return (model, data, train_fun)"""
        args, info = self.load_json(model_folder_name)
        model_name = info["model_name"]
        builder = modelbuilder.get_builder(model_name)
        args.update(new_args)
        model, dataset, train = builder.build(args)
        model_path = self.get_model_load_path(model_folder_name)
        model.load_state_dict(torch.load(model_path, map_location=get_device_label_from_args(args)))
        model.eval()
        return model, dataset, train
