from __future__ import print_function
import torch
import numpy

size_batch = 64
dim_input = 1
dim_output = 1
dim_hidden = 8

model = torch.nn.Sequential(
    torch.nn.Linear(dim_input, dim_hidden),
    torch.nn.ReLU(),
    torch.nn.Linear(dim_hidden, dim_output)
)

fn_loss = torch.nn.MSELoss(reduction="sum")

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


for t in range(10000):
    l = list(range(t // 2, t // 2 + size_batch))
    x = torch.tensor(l, dtype=torch.float).view(size_batch, dim_input)
    y = torch.tensor([2 * n for n in l], dtype=torch.float).view(size_batch, dim_output)

    y_pred = model(x)

    loss = fn_loss(y_pred, y)
    print(t, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("testing model")
for n in range(10):
    print(n, "->", model(torch.tensor([n], dtype=torch.float)))

for n in range(1000, 1010):
    print(n, "->", model(torch.tensor([n], dtype=torch.float)))

for n in range(-110, -100):
    print(n, "->", model(torch.tensor([n], dtype=torch.float)))
