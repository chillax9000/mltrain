from typing import List

import nltk
import torch.utils.data

import mlt.resources.wikipedia as wiki


class Vocab:
    def __init__(self, words):
        self.words = set(words)
        self.empty_token = "[_]"
        self.unknown_token = "[?]"
        self.empty_idx = 0
        self.unknown_idx = 1
        self.itos = dict(enumerate(self.words, start=2))
        self.itos[self.empty_idx] = self.empty_token
        self.itos[self.unknown_idx] = self.unknown_token
        self.stoi = {word: idx for idx, word in self.itos.items()}

    @property
    def size(self):
        return len(self.itos)

    def replace_unknown(self, word_list):
        return [word if word in self.stoi else self.unknown_token for word in word_list]

    def words_to_indexes(self, word_list: List):
        return [self.stoi[word] for word in word_list]

    def indexes_to_words(self, index_list: List):
        return [self.itos[index] for index in index_list]


class TextData:
    def __init__(self, text: str = None, vocab: Vocab = None):
        self.text = text if text is not None else wiki.get_cleaned_text("Cat", wiki.Lang.en).lower()

        nltk.download("punkt")
        nltk_text = nltk.Text(nltk.tokenize.word_tokenize(self.text))

        sentence_detector = nltk.data.load("tokenizers/punkt/english.pickle")
        self.sentences = sentence_detector.tokenize(self.text)

        self.vocab = Vocab(nltk_text.vocab().keys()) if vocab is None else vocab
        self.tokenized_sentences = list(map(self.vocab.replace_unknown, map(nltk.word_tokenize, self.sentences)))

    def get_sentence_indexes(self, n):
        return self.vocab.words_to_indexes(self.tokenized_sentences[n])


class SkipGramDataset(torch.utils.data.Dataset):
    def __init__(self, textdata, device, context_size=2):
        self.textdata = textdata
        self.context_size = context_size
        self.device = device

        self.data = self.build_data(self.textdata)

    def build_data(self, textdata: TextData):
        data = []
        for n in range(len(textdata.sentences)):
            indexes = textdata.get_sentence_indexes(n)
            ngram_position_range = range(-self.context_size, len(indexes))
            ngrams = [self.get_ngram_tensors_at_position(position, indexes) for position in ngram_position_range]
            data.extend(ngrams)
        return data

    def augmented_sentence_indexes(self, indexes):
        extension = [self.textdata.vocab.empty_idx] * self.context_size
        return extension + indexes + extension

    def get_ngram_tensors_at_position(self, position, sentence_indexes):
        augmented_sentence = self.augmented_sentence_indexes(sentence_indexes)
        context_first_position = position + self.context_size
        after_context_position = position + 2 * self.context_size
        return (torch.tensor(augmented_sentence[context_first_position:after_context_position]).to(device=self.device),
                torch.tensor(augmented_sentence[after_context_position]).to(device=self.device))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]
