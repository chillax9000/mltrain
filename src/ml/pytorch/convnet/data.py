import PIL
import numpy as np

import torch


class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, img_width=28, transform=None):
        super().__init__()
        self.x = x
        self.y = y
        self.img_width = img_width
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]

        if self.transform is not None:
            x = 255 * self.transform(PIL.Image.fromarray(np.uint8(x.reshape(self.img_width, self.img_width))))
        else:
            x = torch.Tensor(x).reshape(1, self.img_width, self.img_width)
        return x, y
