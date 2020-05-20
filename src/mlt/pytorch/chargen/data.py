import os
import random
from typing import List

import numpy as np
import torch

import mlt.resources.wikipedia as wiki
import mlt.util.text
from mlt.util import vector

project_path = os.path.dirname(__file__)
default_device = torch.device("cpu")


# Random item from a list
def random_choice(l: List):
    return l[random.randrange(0, len(l))]


# Build the category_lines dictionary, a list of lines per category # well, now words
class Data:
    def __init__(self, device=default_device, all_chars="abcdefghijklmnopqrstuvwxyz"):
        self.device = device

        self.all_chars = all_chars
        self.n_chars = len(self.all_chars) + 1  # + eos

        self.category_lines = self.build_category_lines()
        self.all_categories = list(self.category_lines)
        self.n_categories = len(self.all_categories)

        self.tensors = self.create_tensors()

    def build_category_lines(self):
        raise NotImplementedError

    # Get a random category and random line from that category
    def get_random_training_pair(data):
        category = random_choice(data.all_categories)
        line = random_choice(data.category_lines[category])
        return category, line

    # Store tensors for reuse
    def create_tensors(self, device=None):
        device = device if device is not None else self.device
        tensors = {
            "category": {},
            "char": {}
        }
        for idx, category in enumerate(self.all_categories):
            tensors["category"][category] = torch.Tensor(vector.one_hot(self.n_categories, idx)).to(device=device)
        for idx, char in enumerate(self.all_chars):
            tensors["char"][char] = torch.Tensor(vector.one_hot(self.n_chars, idx)).to(device=device)
        return tensors

    # One-hot vector for category
    def get_category_tensor(self, category):
        """returns a (n_categories, ) tensor"""
        return self.tensors["category"][category]

    def get_char_tensor(self, char):
        return self.tensors["char"][char]

    def get_line_tensor(self, line):
        """returns a (len(line), len(char_tensor)) tensor"""
        return torch.cat(tuple(self.get_char_tensor(c).unsqueeze(0) for c in line))

    def get_char_index(self, char):
        return self.all_chars.find(char)


class DataWord(Data):
    def build_category_lines(self):
        category_lines = {}
        for lang in wiki.Lang:
            words = set()
            for page in wiki.default_pages[lang]:
                text = wiki.get_cleaned_text(page, lang)
                words_ = mlt.util.text.get_words_from_text(text)
                words_ = map(str.lower, map(mlt.util.text.unicode_to_ascii, filter(str.isalpha, words_)))
                words.update(list(set(words_)))
            category_lines[lang] = list(words)

        if not category_lines:
            print("no data...")
            exit(0)
        for cat, words in category_lines.items():
            print(cat, ":", len(words), "words")
        return category_lines


class DataSentence(Data):
    def __init__(self, device=default_device, all_chars="abcdefghijklmnopqrstuvwxyz,.;:?!()-“”'\"`´«»0123456789 "):
        super().__init__(device, all_chars)

    def build_category_lines(self):
        category_lines = {}
        for lang in wiki.Lang:
            sentences = set()
            for page in wiki.default_pages[lang]:
                text = wiki.get_cleaned_text(page, lang)
                sentences_ = list(mlt.util.text.get_sentences_from_text(text))
                n_imported = len(sentences_)
                sentences_ = list(filter(self.acceptable_input,
                                         map(str.lower, map(mlt.util.text.unicode_to_ascii, sentences_))))
                print(n_imported - len(sentences_), "sentences did not meet requirements")
                sentences.update(list(set(sentences_)))
            category_lines[lang] = list(sentences)

        if not category_lines:
            print("no data...")
            exit(0)
        for cat, sentences in category_lines.items():
            print(cat, ":", len(sentences), "sentences of avg len:", int(np.round(np.mean(list(map(len, sentences))))))
        return category_lines

    def acceptable_input(self, s: str):
        return all(c in self.all_chars for c in s)
