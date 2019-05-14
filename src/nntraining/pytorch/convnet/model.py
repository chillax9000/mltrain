import torch
import numpy as np

default_device = "cuda"


def convpool_L_out(L_in, pool_ker, pool_stride, ker, n_layers=1):
    L_in = L_in - n_layers * (ker - 1)
    n = int(np.floor((L_in - (pool_ker - 1) - 1) / pool_stride + 1))
    return n


def convpool_n_param_out(n_chan_out, L_in, pool_ker, pool_stride, ker, n_layers=1):
    return n_chan_out * (convpool_L_out(L_in, pool_ker, pool_stride, ker, n_layers) ** 2)


class ThreeLayersCNN(torch.nn.Module):
    def __init__(self, img_side=28, n_categories=10, ker1=2, ker2=2, pool_ker=2, pool_stride=2, p_dropout=0.2,
                 n_chan1=32, n_chan2=64, ker3=2, n_chan3=128, pool_fn=torch.nn.MaxPool2d, n_linear=128,
                 device=default_device):
        super().__init__()
        self.convnet = torch.nn.Sequential(
            torch.nn.Conv2d(1, n_chan1, ker1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(n_chan1, n_chan1, ker1),
            torch.nn.ReLU(),
            pool_fn(pool_ker, pool_stride),

            torch.nn.Conv2d(n_chan1, n_chan2, ker2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(n_chan2, n_chan2, ker2),
            torch.nn.ReLU(),
            pool_fn(pool_ker, pool_stride),

            torch.nn.Conv2d(n_chan2, n_chan3, ker3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(n_chan3, n_chan3, ker3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(n_chan3, n_chan3, ker3),
            torch.nn.ReLU(),
            pool_fn(pool_ker, pool_stride)
        ).to(device)
        L_l1_out = convpool_L_out(img_side, pool_ker, pool_stride, ker1, n_layers=2)
        L_l2_out = convpool_L_out(L_l1_out, pool_ker, pool_stride, ker2, n_layers=2)
        self.convpool_n_param_out = convpool_n_param_out(n_chan3, L_l2_out, pool_ker, pool_stride, ker3, n_layers=3)

        self.layer_out = torch.nn.Sequential(
            torch.nn.Linear(self.convpool_n_param_out, n_linear),
            torch.nn.ReLU(),
            torch.nn.Dropout2d(p=p_dropout),
            torch.nn.Linear(n_linear, n_linear),
            torch.nn.ReLU(),
            torch.nn.Dropout2d(p=p_dropout),
            torch.nn.Linear(n_linear, n_categories).to(device)
        ).to(device)

    def forward(self, x):
        out = self.convnet(x).reshape(-1, self.convpool_n_param_out)
        out = self.layer_out(out)
        return out
