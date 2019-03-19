import os
import random
from typing import List

import torch

import resources.wikipedia as wiki
import util.text
from util import vector

project_path = os.path.dirname(__file__)
default_device = torch.device("cpu")


# Random item from a list
def random_choice(l: List):
    return l[random.randrange(0, len(l))]


# Build the category_lines dictionary, a list of lines per category # well, now words
class Data:
    def __init__(self, device=default_device, all_chars="abcdefghijklmnopqrstuvwxyz"):
        self.all_chars = all_chars
        self.n_chars = len(self.all_chars) + 1  # + eos
        self.device = device
        self.initialize()
        self.tensors = self.create_tensors()

    def initialize(self):
        self.category_lines = {}
        for lang in wiki.Lang:
            words = set()
            for page in wiki.default_pages[lang]:
                text = wiki.get_cleaned_text(page, lang)
                words_ = util.text.get_words_from_text(text)
                words_ = map(str.lower, map(util.text.unicode_to_ascii, filter(str.isalpha, words_)))
                words.update(list(set(words_)))
            self.category_lines[lang] = list(words)

        if not self.category_lines:
            print("no data...")
            exit(0)
        for cat, words in self.category_lines.items():
            print(cat, ":", len(words), "words")

        self.all_categories = list(self.category_lines)
        self.n_categories = len(self.all_categories)

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