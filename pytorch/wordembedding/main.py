import torch

from pytorch.wordembedding.model import WordEmbSkipGram
from pytorch.wordembedding.train import do_training

model = WordEmbSkipGram(5, 16, 2, 16)
do_training(model, None, n_iter=100, print_every=10)

print(model(torch.tensor([1, 2])).topk(1))
