import collections
import operator
import pickle
import pprint

import simpleclock
import torch
import torch.utils.data


DEVICE = torch.device("cuda")


# will break easily...
class MinMaxList(list):
    def __init__(self):
        super().__init__()
        self._min = float("inf")
        self._max = -float("inf")

    @property
    def min(self):
        return self._min

    @property
    def max(self):
        return self._max

    def append(self, e):
        super().append(e)
        if e > self._max:
            self._max = e
        if e < self._min:
            self._min = e


class TrainInfo:
    def __init__(self, valid={}, train={}):
        self.valid = collections.defaultdict(MinMaxList)
        self.valid.update(valid)
        self.train = collections.defaultdict(MinMaxList)
        self.train.update(train)

    def save(self, path):
        packed = {
            "valid": dict(self.valid),
            "train": dict(self.train),
        }
        with open(path, "wb") as f:
            pickle.dump(packed, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            packed = pickle.load(f)
            return cls(valid=packed["valid"],
                       train=packed["train"])

    @staticmethod
    def _dict_to_repr(d):
        return dict(map(lambda k_v: (k_v[0], f"{len(k_v[1])} elements"), d.items()))

    def __repr__(self):
        return pprint.pformat({"valid": self._dict_to_repr(self.valid),
                               "train": self._dict_to_repr(self.train), })


class Measurer:
    def update(self, batch, get_input, get_target, batch_output, batch_loss):
        raise NotImplementedError

    def to_dict(self):
        raise NotImplementedError

    @classmethod
    def getter(cls, *args, **kwargs):
        return lambda: cls(*args, **kwargs)


class AccLossMeasurer(Measurer):
    def __init__(self, fun_accuracy):
        super().__init__()
        self.loss = 0
        self.acc = 0
        self._count = 0

        self.fun_accuracy = fun_accuracy

    def update(self, batch, get_input, get_target, batch_output, batch_loss):
        self.loss += batch_loss.item()
        self.acc += self.fun_accuracy(batch_output, get_target(batch)).item()
        self._count += 1

    def to_dict(self):
        return {
            "acc": self.acc / self._count,
            "loss": self.loss / self._count,
        }


def train(model, iterator, optimizer, criterion, get_input, get_target, measurer):
    model.train()

    for batch in iterator:
        optimizer.zero_grad()
        output = model(get_input(batch)).squeeze(1)  # rm squeeze ?
        loss = criterion(output, get_target(batch))
        loss.backward()
        optimizer.step()

        if measurer:
            measurer.update(batch, get_input, get_target, output, loss)

    return measurer.to_dict()


def evaluate(model, iterator, criterion, get_input, get_target, measurer):
    model.eval()

    with torch.no_grad():
        for batch in iterator:
            output = model(get_input(batch)).squeeze(1)  # rm squeeze ?
            loss = criterion(output, get_target(batch))

        if measurer:
            measurer.update(batch, get_input, get_target, output, loss)

    return measurer.to_dict()


def update_traininfo_from_dict(traininfo_attr, dict_):
    for k, v in dict_.items():
        traininfo_attr[k].append(v)


def print_status(epoch, n_epochs, train_measures, valid_measures, is_best, duration):
    print("Epoch: {e:.<{max_len}}. T, V acc: {train:.1f}%, {valid:.1f}%. Took {t:.2f}s.{best}"
          .format(e=epoch + 1,
                  max_len=len(str(n_epochs)),
                  train=100 * train_measures["acc"],
                  valid=100 * valid_measures["acc"],
                  t=duration,
                  best=" (+)" if is_best else ""))


def is_max_of(metric_name):
    def fun(measures, training_info_sub):
        return measures[metric_name] > training_info_sub[metric_name].max
    return fun


def is_min_of(metric_name):
    def fun(measures, training_info_sub):
        return measures[metric_name] < training_info_sub[metric_name].min
    return fun


def do_training(model, name, iter_train, iter_valid, optimizer, criterion, n_epochs,
                fun_train=train, fun_eval=evaluate,
                training_info=None, fun_stop=lambda: False, fun_save=None,
                get_input=operator.attrgetter("input"),
                get_target=operator.attrgetter("target"),
                measurer_getter=None,
                fun_is_best=lambda *a, **kwa: False,
                fun_print_status=print_status):
    if fun_save is None:
        print("Not saving best model.")
    if measurer_getter is None:
        print("No metrics computed (measurer is missing).")

    clock = simpleclock.Clock.started()
    torch.cuda.empty_cache()
    training_info = training_info if training_info is not None else TrainInfo()

    print(f"Starting training: {n_epochs} epochs.")
    for epoch in range(n_epochs):
        if fun_stop():
            print("Interrupted. (fun_stop)")
            break

        clock.elapsed_since_start.call()  # meh

        train_measures = fun_train(model, iter_train, optimizer, criterion, get_input, get_target, measurer_getter())
        valid_measures = fun_eval(model, iter_valid, criterion, get_input, get_target, measurer_getter())

        if measurer_getter is not None:
            _is_best = fun_is_best(valid_measures, training_info.valid)
            fun_print_status(epoch, n_epochs, train_measures, valid_measures, _is_best, clock.elapsed_since_last_call())

            update_traininfo_from_dict(training_info.train, train_measures)
            update_traininfo_from_dict(training_info.valid, valid_measures)

            if _is_best:
                if fun_save is not None:
                    fun_save(model)
        else:
            print("Epoch: {e:.<{max_len}}. Took {t:.2f}s."
                  .format(e=epoch + 1, max_len=len(str(n_epochs)), t=clock.elapsed_since_last_call()))

    clock.elapsed_since_start.print(f"Trained {name}, {n_epochs} epochs, for")
    return training_info
