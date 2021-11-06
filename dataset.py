"""
dataset.py
"""
import random
import torch
import torch.nn as nn
import os
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict


"""
1. Download 14 videos
2. Downsample from 398x224 at 25fps to 112x112 at 25fps
3. DataSet (this file)
    - __getitem__(self) - returns one 0.5 s block: output.shape = (3 x 60 (60s clip sampled every second) x 112 x 112)
"""
class PretrainDataset(Dataset):
    
    def __init__(self, csv_path, data_path, args, train=True, transform=None):

    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):

