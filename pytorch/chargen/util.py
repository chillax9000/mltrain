import glob
import os
import string
import unicodedata
from io import open

all_letters = string.ascii_letters
n_letters = len(all_letters) + 1  # + eos

project_path = os.path.dirname(__file__)


def findFiles(path):
    return glob.glob(path)


# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s: str):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


# Build the category_lines dictionary, a list of lines per category
with open(os.path.join(project_path, "Chat")) as f:
    text = f.read()
text = text.replace("  ", " ")
text = text.replace(" ", "\n")
words = text.split("\n")
words = map(str.lower, map(unicodeToAscii, filter(str.isalpha, words)))
words = list(set(words))

category_lines = {"fr": words}

all_categories = list(category_lines)
n_categories = len(all_categories)
