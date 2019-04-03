import torch

from nntraining.pytorch.wordembedding.data import SkipGramDataset, TextData

text = """
The sky is blue.
"""


def test_skipgramdataset():
    textdata = TextData(text)
    empty = textdata.vocab.empty_token
    expected_triplets = [
        [empty, empty, "The"],
        [empty, "The", "sky"],
        ["The", "sky", "is"],
        ["sky", "is", "blue"],
        ["is", "blue", "."],
        ["blue", ".", empty],
        [".", empty, empty],
    ]
    dataset = SkipGramDataset(textdata, device=torch.device("cpu"))
    for sample, triplet in zip(dataset, expected_triplets):
        (context, last) = sample
        sample_as_strings = textdata.vocab.indexes_to_words(context.tolist() + [last.tolist()])
        assert sample_as_strings == triplet
