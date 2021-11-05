"""
model.py
"""

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class VideoFeatureBackbone(nn.module):
    def __init__(self):
        super(VideoFeatureBackbone, self).__init__()
        self.resnet3d = models.video.r3d_18(pretrained=True)
        self.resnet3d = self.resnet3d.features[:18]
        for idx in range(13):
            for param in self.resnet3d[idx].parameters():
                param.requires_grad = False

    def forward(self, x):
    y = self.resnet3d(x)
    return y

class ConvClassifier(nn.module):
    def __init__(self, backbone):
        super(ConvClassifier, self).__init__()
        if backbone == 'resnet3d':
            self.backbone = VideoFeatureBackbone()
        