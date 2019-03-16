import torch

from pytorch.chargen.train import get_input_tensor, get_category_tensor


# Sample from a category and starting letter
def sample(rnn, data, category, start_letter='A', max_length=20):
    with torch.no_grad():  # no need to track history in sampling
        category_tensor = get_category_tensor(data, category, rnn.device)
        input = get_input_tensor(data, start_letter, rnn.device)
        hidden = rnn.initHidden()

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
            input = get_input_tensor(data, letter, rnn.device)

        return output_name


# Get multiple samples from one category and multiple starting letters
def samples(rnn, data, category, start_letters='ABC'):
    for start_letter in start_letters:
        print(sample(rnn, data, category, start_letter))
