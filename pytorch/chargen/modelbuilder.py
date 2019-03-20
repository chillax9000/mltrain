from pytorch.chargen import train, data, model
from command import CmdArg
from pytorch.device import get_device_from_args


class ModelBuilder:
    def __init__(self, model_class, model_feeder, data_class, data_feeder, train_fun):
        self.model_class = model_class
        self.model_feeder = model_feeder
        self.data_class = data_class
        self.data_feeder = data_feeder
        self.fun_train = train_fun

    @staticmethod
    def feed(eater, food):
        if isinstance(food, tuple):
            args, kwargs = food
            return eater(*args, **kwargs)
        elif isinstance(food, dict):
            return eater(**food)

    def build(self, args):
        data = self.feed(self.data_class, self.data_feeder(args))
        model = self.feed(self.model_class, self.model_feeder(args, data))
        return model, data, self.fun_train


MODELS = {}


def add_model(name, builder):
    MODELS[name] = builder


def get_model(name):
    return MODELS.get(name, None)


add_model("rnn-simple_words",
          ModelBuilder(model_class=model.SimpleRNN,
                       model_feeder=lambda args, data: ((data.n_chars, data.n_categories, args[CmdArg.hidden]),
                                                        {"device": get_device_from_args(args)}),
                       data_class=data.DataWord,
                       data_feeder=lambda args: ((get_device_from_args(args),), {}),
                       train_fun=train.train_nn_rnn))

add_model("rnn_words",
          ModelBuilder(model_class=model.RNN,
                       model_feeder=lambda args, data: (
                       (data.n_chars, args[CmdArg.hidden], data.n_chars, data.n_categories),
                       {"device": get_device_from_args(args)}),
                       data_class=data.DataWord,
                       data_feeder=lambda args: ((get_device_from_args(args),), {}),
                       train_fun=train.train))
