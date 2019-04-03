import os

import torch

import nntraining.pytorch.generic
from nntraining.pytorch.wordembedding.data import SkipGramDataset, TextData, Vocab
from nntraining.pytorch.wordembedding.model import WordEmbSkipGram

data = TextData()
dataset = SkipGramDataset(data, torch.device("cpu"))
model = WordEmbSkipGram(vocab_size=data.vocab.size,
                        embedding_dim=16,
                        context_size=2,
                        hidden_layer_size=16)
nntraining.pytorch.generic.do_training(model, dataset, nntraining.pytorch.generic.train, os.path.dirname(__file__),
                                       n_iter=5000, print_every=1000)

max_words = 16
sentence_number = 42
print(data.sentences[sentence_number])
context = data.get_sentence_indexes(sentence_number)[:2]
prediction_index = None
generated_indexes = context
n_words = 0
while prediction_index != data.empty_token_idx and n_words < max_words:
    prediction_index = model(torch.tensor([context])).topk(1)[1].item()
    generated_indexes += [prediction_index]
    context = generated_indexes[-2:]
    n_words += 1

print(" ".join(data.indexes_to_words(generated_indexes)))
