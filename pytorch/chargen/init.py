import os

import resources.wikipedia as wiki
import util.text

all_letters = "abcdefghijklmnopqrstuvwxyz"
n_letters = len(all_letters) + 1  # + eos

project_path = os.path.dirname(__file__)

# Build the category_lines dictionary, a list of lines per category
words = set()
for page in wiki.default_pages[wiki.Lang.fr]:
    text = wiki.get_cleaned_text(page, wiki.Lang.fr)
    words_ = util.text.get_words_from_text(text)
    words_ = map(str.lower, map(util.text.unicode_to_ascii, filter(str.isalpha, words_)))
    words.update(list(set(words_)))
category_lines = {"fr": list(words)}
for cat, words in category_lines.items():
    print(cat, ":", len(words), "words")
    print(sorted(category_lines["fr"], key=len))

all_categories = list(category_lines)
n_categories = len(all_categories)
