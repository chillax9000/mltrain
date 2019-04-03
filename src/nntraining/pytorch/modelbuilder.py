import nntraining.pytorch.generic
import nntraining.pytorch.chargen.model
import nntraining.pytorch.chargen.data
import nntraining.pytorch.chargen.train
import nntraining.pytorch.wordembedding.model
import nntraining.pytorch.wordembedding.data
from nntraining.command import CmdArg
from nntraining.pytorch.device import get_device_from_args


class ModelBuilder:
    def __init__(self, model_class, model_feeder, data_class, data_feeder, dataset_class=None, dataset_feeder=None,
                 train_fun=nntraining.pytorch.generic.train, serialize_name=None):
        self.model_class = model_class
        self.model_feeder = model_feeder
        self.data_class = data_class
        self.data_feeder = data_feeder
        self.dataset_class = dataset_class
        self.dataset_feeder = dataset_feeder
        self.fun_train = train_fun

        self._serialize_name = serialize_name

    @staticmethod
    def feed(eater, food):
        """food to be either ((args), {kwargs}) or (args)"""
        if isinstance(food, tuple):
            args, kwargs = food
            return eater(*args, **kwargs)
        elif isinstance(food, dict):
            return eater(**food)

    def build(self, args):
        data = self.feed(self.data_class, self.data_feeder(args))
        model = self.feed(self.model_class, self.model_feeder(args, data))
        dataset = self.feed(self.dataset_class, self.dataset_feeder(args, data))

        model._serialize_name = self._serialize_name

        return model, dataset, self.fun_train


MODELS = {}


def add_model(name, builder):
    MODELS[name] = builder
    builder._serialize_name = name


def get_builder(name):
    builder = MODELS.get(name, None)
    if builder is None:
        raise ValueError(f"Could not find a model named: {name}")
    return builder


def print_model_list():
    if MODELS:
        print("Available models:")
        for model_name in MODELS:
            print("+", model_name)
    else:
        print("No model available")


def build_from_args(args):
    """returns: model, data, train_fun"""
    if not args[CmdArg.model]:
        raise ValueError(f"A model is required, specify it with option {CmdArg.model.cmd_name}")
    builder = get_builder(args[CmdArg.model])
    return builder.build(args)


# MODELS #

add_model("rnn-simple_words",
          ModelBuilder(model_class=nntraining.pytorch.chargen.model.SimpleRNN,
                       model_feeder=lambda args, data: ((data.n_chars, data.n_categories, args[CmdArg.hidden]),
                                                        {"device": get_device_from_args(args)}),
                       data_class=nntraining.pytorch.chargen.data.DataWord,
                       data_feeder=lambda args: ((get_device_from_args(args),), {}),
                       train_fun=nntraining.pytorch.chargen.train.train_nn_rnn)
          )

add_model("rnn_words",
          ModelBuilder(model_class=nntraining.pytorch.chargen.model.RNN,
                       model_feeder=lambda args, data: (
                           (data.n_chars, args[CmdArg.hidden], data.n_chars, data.n_categories),
                           {"device": get_device_from_args(args)}),
                       data_class=nntraining.pytorch.chargen.data.DataWord,
                       data_feeder=lambda args: ((get_device_from_args(args),), {}),
                       train_fun=nntraining.pytorch.chargen.train.train)
          )

add_model("word-embed-skipgram",
          ModelBuilder(model_class=nntraining.pytorch.wordembedding.model.WordEmbSkipGram,
                       model_feeder=lambda args, data:(
                           (data.vocab.size, args[CmdArg.embedding], args[CmdArg.context], args[CmdArg.hidden]),
                           {"device": get_device_from_args(args)}),
                       data_class=nntraining.pytorch.wordembedding.data.TextData,
                       data_feeder=lambda args: ((), {}),
                       dataset_class=nntraining.pytorch.wordembedding.data.SkipGramDataset,
                       dataset_feeder=lambda args, data: ((data, ), {"device": get_device_from_args(args)}))
          )
