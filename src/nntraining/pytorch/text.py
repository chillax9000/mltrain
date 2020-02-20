import torch


DEVICE = torch.device("cuda")


def predict_tokens(tokens, model, stoi, device=DEVICE):
    model.eval()
    idxs = [stoi[t] for t in tokens]
    inp = torch.LongTensor(idxs).reshape(-1, 1).to(device)
    output = model((inp, torch.LongTensor([len(tokens)])))
    return output.item()


def predict(sentence, model, stoi, tokenizer):
    return predict_tokens(list(map(str, tokenizer(sentence))), model, stoi)


class Predictor:
    def __init__(self, model, stoi, tokenizer, device=DEVICE):
        self.model = model
        self.stoi = stoi
        self.tokenizer = tokenizer
        self.device = device

    def predict(self, sentence):
        return predict(sentence, self.model, self.stoi, self.tokenizer)

    def predict_tokens(self, tokens):
        return predict_tokens(tokens, self.model, self.stoi, self.device)

    def __call__(self, sentence):
        return self.predict(sentence)
