"""
dataset.py
"""
import pickle
import random
import torch
import torch.nn as nn
import os
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict
import torchvision.transforms
from torchvision import transforms as T

"""
1. Download 14 videos
2. Downsample from 398x224 at 25fps to 112x112 at 25fps
3. DataSet (this file)
    - __getitem__(self) - returns one 60s block: output.shape = (3 x 60 (60s clip sampled every second) x 112 x 112)
"""

class SoccerDataset(Dataset):
    
    def __init__(self, data_path, clip_size, transform=None):
        self.data_path = data_path
        self.clip_size = clip_size
        tensor_path = data_path + f"blocks_{clip_size}.pt"
        self.input = torch.load(tensor_path)
        label_path = data_path + f"block_num_to_label_{clip_size}.pt"
        self.labels = torch.load(label_path)
        if not transform:
            normalize = T.Normalize((0.43216,0.394666, 0.37645), (0.22803, 0.22145, 0.216989))
            self.transform = T.Compose([normalize])
        else:
            self.transform = transform

    def __len__(self):
        return self.input.shape[0]
    
    def __getitem__(self, idx):
        return self.transform(self.input[idx]).permute(1, 0, 2, 3), self.labels[idx]

        
class ToyDataset(Dataset):
    def __init__(self, length, num_classes, transform=None):
        self.length = length
        self.num_classes = num_classes
        # self.dist = torch.distributions.Bernoulli(torch.tensor([alpha]))
        if transform:
            self.transform = transform

    def __len__(self):
        return self.length 
    
    def __getitem__(self, idx):
        imgs = torch.rand(60, 3, 112, 112) * 255
        label = int(torch.randint(self.num_classes, (1,)))
        if self.transform:
            imgs = self.transform(imgs)
        return imgs.permute(1, 0, 2, 3), 7
