import torch

from ml.pytorch.chargen.train import get_input_from_category_and_line_tensors


# Sample from a category and starting char
def sample(rnn, data, category, start_char, max_length=64):
    with torch.no_grad():  # no need to track history in sampling
        category_tensor = data.get_category_tensor(category)
        input = data.get_line_tensor(start_char)
        hidden = rnn.init_hidden()

        output_name = start_char

        for i in range(max_length):
            output, hidden = rnn(category_tensor, input[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0]
            if topi == data.n_chars - 1:
                break
            else:
                char = data.all_chars[topi]
                output_name += char
            input = data.get_line_tensor(start_char)

        return output_name


# Get multiple samples from one category and multiple starting chars
def samples(rnn, data, category, start_chars):
    for start_char in start_chars:
        print(sample(rnn, data, category, start_char))


def sample_nn_rnn(rnn, data, category, start_char, max_length=64):
    with torch.no_grad():  # no need to track history in sampling
        category_tensor = data.get_category_tensor(category)
        line_tensor = data.get_line_tensor(start_char)
        output_name = start_char

        hidden = None

        for i in range(max_length):
            input = get_input_from_category_and_line_tensors(category_tensor, line_tensor)
            output, hidden = rnn(input, hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == data.n_chars - 1:
                break
            else:
                char = data.all_chars[topi]
                output_name += char
            line_tensor = data.get_line_tensor(char)

        return output_name


def samples_nn_rnn(rnn, data, category, start_chars):
    for start_char in start_chars:
        print(sample_nn_rnn(rnn, data, category, start_char))
