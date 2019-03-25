import torch

from pytorch.wordembedding.model import WordEmbSkipGram
from pytorch.wordembedding.train import do_training
from pytorch.wordembedding.data import Data

data = Data()
model = WordEmbSkipGram(vocab_size=data.vocab_size,
                        embedding_dim=16,
                        context_size=2,
                        hidden_layer_size=16)
do_training(model, data, n_iter=1000, print_every=100)

print(model(torch.tensor([1, 2])).topk(1))
