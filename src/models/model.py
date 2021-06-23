import torch
import torchvision.models as models
from torch import nn


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Pretrained Model
        self.network = models.resnet50(pretrained=True)
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 6)

    def forward(self, xb):
        return torch.sigmoid(self.network(xb))
