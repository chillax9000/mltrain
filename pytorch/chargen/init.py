import os
import unicodedata

import resources.wikipedia

all_letters = "abcdefghijklmnopqrstuvwxyz"
n_letters = len(all_letters) + 1  # + eos

project_path = os.path.dirname(__file__)


# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s: str):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


# Build the category_lines dictionary, a list of lines per category
text = resources.wikipedia.get_cleaned_text("Chat", resources.wikipedia.Lang.fr)
text = text.replace("  ", " ")
text = text.replace(" ", "\n")
words = text.split("\n")
words = map(str.lower, map(unicodeToAscii, filter(str.isalpha, words)))
words = list(set(words))

category_lines = {"fr": words}

all_categories = list(category_lines)
n_categories = len(all_categories)
