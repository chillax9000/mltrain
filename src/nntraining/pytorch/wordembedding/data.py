from typing import List

import nltk
import torch.utils.data

import nntraining.resources.wikipedia as wiki


class TextData:
    def __init__(self, text: str = None):
        self.text = text if text is not None else wiki.get_cleaned_text("Cat", wiki.Lang.en).lower()

        nltk.download("punkt")
        nltk_text = nltk.Text(nltk.tokenize.word_tokenize(self.text))

        sentence_detector = nltk.data.load("tokenizers/punkt/english.pickle")
        self.sentences = sentence_detector.tokenize(self.text)
        self.tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in self.sentences]

        self.words = sorted(list(set(nltk_text.vocab().keys())))
        self.vocab_size = len(self.words) + 1  # adding an empty token

        # idx: word
        self.empty_token_idx = 0
        self.idx_word_dict = dict(map(lambda idx_word: (idx_word[0] + 1, idx_word[1]), enumerate(self.words)))
        self.idx_word_dict[self.empty_token_idx] = "[EMPTY TOKEN]"

        self.word_idx_dict = {word: idx for idx, word in self.idx_word_dict.items()}  # word: idx

    def words_to_indexes(self, word_list: List):
        return [self.word_idx_dict[word] for word in word_list]

    def indexes_to_words(self, index_list: List):
        return [self.idx_word_dict[index] for index in index_list]

    def get_sentence_indexes(self, n):
        return self.words_to_indexes(self.tokenized_sentences[n])


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
        extension = [self.textdata.empty_token_idx] * self.context_size
        return extension + indexes + extension

    def get_ngram_tensors_at_position(self, position, sentence_indexes):
        augmented_sentence = self.augmented_sentence_indexes(sentence_indexes)
        return (torch.tensor(augmented_sentence[position + self.context_size: position + 2 * self.context_size]
                             ).to(device=self.device),
                torch.tensor(augmented_sentence[position + 2 * self.context_size]).to(device=self.device))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]
