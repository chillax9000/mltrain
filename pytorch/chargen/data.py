import os

import resources.wikipedia as wiki
import util.text

project_path = os.path.dirname(__file__)


# Build the category_lines dictionary, a list of lines per category # well, now words
class Data:
    def __init__(self):
        self.all_letters = "abcdefghijklmnopqrstuvwxyz"
        self.n_letters = len(self.all_letters) + 1  # + eos

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
