"""
model.py
"""

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class VideoFeatureBackbone(nn.Module):
    def __init__(self):
        super(VideoFeatureBackbone, self).__init__()
        self.resnet3d = models.video.r3d_18(pretrained=True)
        # Remove last classification layer -> num_features = 512
        self.resnet3d = torch.nn.Sequential(*(list(self.resnet3d.children())[:-1]))
        # Freezing inner layers
        # for i, param in enumerate(self.resnet3d.parameters()):
        #     if i == 10:
        #         break
        #     param.requires_grad = False

    def forward(self, x):
        y = self.resnet3d(x)
        return y

class ConvClassifier(nn.Module):
    def __init__(self, backbone, num_classes):
        super(ConvClassifier, self).__init__()
        if backbone == 'resnet3d':
            self.backbone = VideoFeatureBackbone()
        else: 
            raise Exception('No backbone selected!')
        self.model = nn.Sequential(*(list(self.backbone.children())[:]), nn.Flatten(), nn.Linear(512, num_classes), nn.Softmax(dim=1))

    def forward(self, x):
        return self.model(x)

class ConvLSTMClassifier(nn.Module):
    def __init__(self, backbone, hidden_size, num_layers, num_classes):
        super(ConvLSTMClassifier, self).__init__()
        if backbone == 'resnet3d':
            self.backbone = VideoFeatureBackbone()
        else: 
            raise Exception('No backbone selected!')
        self.flatten = nn.Flatten()
        self.num_classes = num_classes
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_size, num_layers=num_layers)
        self.linear = nn.Linear(hidden_size, num_classes)
        # self.out = nn.Sigmoid()
        # self.hn = torch.zeros(num_layers, 1, hidden_size)
        # self.cn = torch.zeros(num_layers, 1, hidden_size)

    def forward(self, x):
        # breakpoint()
        b, c, t, h, w = x.shape
        logits = torch.zeros(b, self.num_classes)
        for i in range(b):
            out = self.backbone(x[i,].unsqueeze(0))
            out = self.flatten(out)
            out, _ = self.lstm(out.unsqueeze(1))
            out = self.linear(out)
            logits[i,:] = out[0,0,:]
        return logits
        
        