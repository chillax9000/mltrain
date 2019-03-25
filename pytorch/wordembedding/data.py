import string
from typing import List

import nltk
import numpy as np

import resources.wikipedia as wiki


class Data:
    chars = string.ascii_letters

    text = wiki.get_cleaned_text("Cat", wiki.Lang.en)
    text = text.lower()

    nltk.download("punkt")
    nltk_text = nltk.Text(nltk.tokenize.word_tokenize(text))

    sentence_detector = nltk.data.load("tokenizers/punkt/english.pickle")
    sentences = sentence_detector.tokenize(text)

    # extract sequences of words
    words = sorted(list(set(nltk_text.vocab().keys())))
    print(words)

    vocab_size = len(words) + 1  # adding an empty token

    idx_word_dict = dict(enumerate(words))  # idx: word
    id_empty_token = vocab_size - 1
    word_idx_dict = {word: idx for idx, word in idx_word_dict.items()}  # word: idx

    def words_to_indexes(self, word_list: List):
        return [self.word_idx_dict[word] for word in word_list]

    def get_trigram_indexes(self):
        n = np.random.randint(len(self.sentences))
        sentence = self.sentences[n]
        tokens = nltk.word_tokenize(sentence)
        indexes = self.words_to_indexes(tokens)
        m = np.random.randint(len(indexes))
        indexes_augmented = [self.id_empty_token] * 3 + indexes + [self.id_empty_token] * 3

        selection =  indexes_augmented[m: m + 3]
        return selection