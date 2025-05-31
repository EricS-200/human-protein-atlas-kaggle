from torch.utils.data import Dataset, DataLoader
import torch
from torchvision.io import read_image
import numpy as np

class DataOrganizer(Dataset): 
    def __init__(self, ids, directory, labels=None, transform=False):
        self.ids = ids 
        self.labels = labels
        self.directory = directory
        self.transform = transform

    def getPath(self, idx):
        base = f"{self.directory}/{self.ids[idx]}"
        paths = [f"{base}_{color}.png" for color in ["green", "blue", "red", "yellow"]]
        return paths
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        paths = self.getPath(idx)
        img_tensors = [read_image(p) for p in paths]
        x = torch.cat(img_tensors, dim=0).float()
        x.div_(255.)
        if self.transform: # data augmentations and transformations passed in
            x = self.transform(x)
        if self.labels is None:
            return x
        y = torch.from_numpy(self.labels[idx]).float()
        return x, y # return a tensor w/ 4 channels each 512x512: numeric representation of the 4 channel image, and the associated label
    