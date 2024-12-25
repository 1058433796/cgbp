import torch
from torch import nn
from torchvision import models
class MyResnet(nn.Module):
    def __init__(self, num_class, load_weights=True):
        super(MyResnet, self).__init__()
        self.num_class = num_class
        self.model = models.resnet50(weights='IMAGENET1K_V1')
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_class)
        )
        if load_weights:
            self.load_state_dict(torch.load(r'C:\Users\cxs\Desktop\FaithfulGradient\models\myresnet.pth'))

    def forward(self, x):
        x = self.model(x)
        return x