from typing import List

from torch import nn

import resources.wikipedia as wiki
import string
import util.text
import nltk
chars = string.ascii_letters

text = wiki.get_cleaned_text("Cat", wiki.Lang.en)

nltk.download("punkt")
text = nltk.Text(nltk.tokenize.word_tokenize(text))
print(text)

# todo list
# extract sequences of words
words = sorted(list(set(map(str.lower, text.vocab().keys()))))
print(words)

idx_word_dict = dict(enumerate(words))  # idx: word
word_idx_dict = {word: idx for idx, word in idx_word_dict.items()}  # word: idx

# create embedding
## -> to model

##


def words_to_indexes(word_list: List):
    return [word_idx_dict[word] for word in word_list]
