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
        # Remove last classification layer -> num_features = 512
        self.resnet3d = torch.nn.Sequential(*(list(self.resnet3d.children())[:-1]))
        # Freezing inner layers
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
        else: 
            raise Exception('No backbone selected!')
        self.model = nn.Sequential(*(self.backbone.modules), nn.Linear(512, 2))

    def forward(self, x):
        return self.model(x)

        
        
        
        