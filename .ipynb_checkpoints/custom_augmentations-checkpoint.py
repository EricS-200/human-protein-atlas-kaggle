import random
import torch

class RandomRotate90:
    def __call__(self, x):
        k = random.randint(0, 3)
        return torch.rot90(x, k, [1, 2])

class FourChannelRandomJitter:
    def __init__(self, factor=0.1):
        self.factor = factor

    def __call__(self, img):
        for i in range(4):
            jitter = 1.0 + (torch.rand(1).item() - 0.5) * 2 * self.factor
            img[i] = img[i] * jitter
        return img