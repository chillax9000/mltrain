import torch

from pytorch.chargen.train import get_input_tensor, get_category_tensor, create_tensors


# Sample from a category and starting letter
def sample(rnn, data, category, start_letter, max_length=20):
    tensors = create_tensors(data, rnn.device)
    with torch.no_grad():  # no need to track history in sampling
        category_tensor = get_category_tensor(tensors, category)
        input = get_input_tensor(tensors, start_letter)
        hidden = rnn.init_hidden()

        output_name = start_letter

        for i in range(max_length):
            output, hidden = rnn(category_tensor, input[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == data.n_letters - 1:
                break
            else:
                letter = data.all_letters[topi]
                output_name += letter
            input = get_input_tensor(tensors, letter)

        return output_name


# Get multiple samples from one category and multiple starting letters
def samples(rnn, data, category, start_letters):
    for start_letter in start_letters:
        print(sample(rnn, data, category, start_letter))


def sample_nn_rnn(rnn, data, category, start_letter, max_length=20):
    tensors = create_tensors(data, rnn.device)
    with torch.no_grad():  # no need to track history in sampling
        category_tensor = get_category_tensor(tensors, category)
        input = get_input_tensor(tensors, start_letter)
        output_name = start_letter
        output, hidden = rnn(torch.cat((category_tensor.unsqueeze(0), input), dim=2))

        for i in range(max_length):
            output, hidden = rnn(torch.cat((category_tensor.unsqueeze(0), input), dim=2), hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == data.n_letters - 1:
                break
            else:
                letter = data.all_letters[topi]
                output_name += letter
            input = get_input_tensor(tensors, letter)

        return output_name


def samples_nn_rnn(rnn, data, category, start_letters):
    for start_letter in start_letters:
        print(sample_nn_rnn(rnn, data, category, start_letter))
