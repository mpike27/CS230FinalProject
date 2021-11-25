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

"""
1. Download 14 videos
2. Downsample from 398x224 at 25fps to 112x112 at 25fps
3. DataSet (this file)
    - __getitem__(self) - returns one 60s block: output.shape = (3 x 60 (60s clip sampled every second) x 112 x 112)
"""
class SoccerDataset(Dataset):
    
    def __init__(self, data_path):
        self.data_path = data_path
        # self.labels_path = data_path
        self.args = args
        self.train = train
        self.labels = json.load(data_path)
        if not transform:
            normalize = transforms.Normalize((0.43216,0.394666, 0.37645), (0.22803, 0.22145, 0.216989))
            self.transform = transforms.Compose([normalize])

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        video_path = os.path.join(self.data_path, "/Data/SoccerNet/england_epl")
        save_path = os.path.join(self.data_path, "/Data/SoccerNet/Tensors")
        with open(os.path.join(save_path, "block_num_to_label.pkl"), "rb") as fd:
            labels = pickle.load(fd)
        return labels[idx], os.path.join(self.data_path, f"/Data/SoccerNet/Tensors/block_{idx}.pt")

        
class ToyDataset(Dataset):
    def __init__(self, length, alpha, transform=None):
        self.length = length
        self.dist = torch.distributions.Bernoulli(torch.tensor([alpha]))
        if transform:
            self.transform = transform

    def __len__(self):
        return self.length 
    
    def __getitem__(self, idx):
        imgs = torch.rand(60, 3, 112, 112) * 255
        label = self.dist.sample()
        if self.transform:
            imgs = self.transform(imgs)
        return imgs.permute(1, 0, 2, 3), label
