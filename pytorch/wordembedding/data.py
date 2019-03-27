import string
import torch.utils.data
from typing import List

import nltk
import numpy as np

import resources.wikipedia as wiki


class TextData:
    chars = string.ascii_letters

    text = wiki.get_cleaned_text("Cat", wiki.Lang.en)
    text = text.lower()

    nltk.download("punkt")
    nltk_text = nltk.Text(nltk.tokenize.word_tokenize(text))

    sentence_detector = nltk.data.load("tokenizers/punkt/english.pickle")
    sentences = sentence_detector.tokenize(text)

    words = sorted(list(set(nltk_text.vocab().keys())))
    vocab_size = len(words) + 1  # adding an empty token

    idx_word_dict = dict(enumerate(words))  # idx: word
    index_empty_token = vocab_size - 1
    idx_word_dict[index_empty_token] = "[EMPTY TOKEN]"

    word_idx_dict = {word: idx for idx, word in idx_word_dict.items()}  # word: idx

    def words_to_indexes(self, word_list: List):
        return [self.word_idx_dict[word] for word in word_list]

    def indexes_to_words(self, index_list: List):
        return [self.idx_word_dict[index] for index in index_list]

    def get_sentence_indexes(self, n):
        sentence = self.sentences[n]
        tokens = nltk.word_tokenize(sentence)
        return self.words_to_indexes(tokens)

    def get_trigram_indexes(self):
        n = np.random.randint(len(self.sentences))
        indexes = self.get_sentence_indexes(n)
        m = np.random.randint(len(indexes))
        indexes_augmented = [self.index_empty_token] * 3 + indexes + [self.index_empty_token] * 3

        selection = indexes_augmented[m: m + 3]
        return selection


class SkipGramDataset(torch.utils.data.Dataset):
    def __init__(self, textdata, context_size=2):
        self.textdata = textdata
        self.context_size = context_size

        self.data = self.build_data(self.textdata)

    def build_data(self, textdata):
        data = []
        for n in range(len(textdata.sentences)):
            indexes = textdata.get_sentence_indexes(n)
            ngram_position_range = range(-self.context_size, len(indexes))
            ngrams = [self.get_ngram_tensors_at_position(position, indexes) for position in ngram_position_range]
            data.extend(ngrams)
        return data

    def augmented_sentence_indexes(self, indexes):
        extension = [self.textdata.index_empty_token] * self.context_size
        return extension + indexes + extension

    def get_ngram_tensors_at_position(self, position, sentence_indexes):
        augmented_sentence = self.augmented_sentence_indexes(sentence_indexes)
        return (torch.tensor(augmented_sentence[position + self.context_size: position + 2 * self.context_size]),
                torch.tensor(augmented_sentence[position + 2 * self.context_size]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]
