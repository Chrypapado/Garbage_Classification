import torch
from torch import nn
import torch.nn.functional as F

class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=5, kernel_size=5)
        self.fc1 = nn.Linear(in_features=5*40*55, out_features=1000)
        self.fc2 = nn.Linear(in_features=1000, out_features=100)
        self.fc3 = nn.Linear(in_features=100, out_features=6)

        self.maxpool = nn.MaxPool2d(3, 3)
        self.leakyrelu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.1)        

    def forward(self, x):
        x = self.maxpool(self.leakyrelu(self.conv1(x)))
        x = self.maxpool(self.leakyrelu(self.conv2(x)))
        x = x.view(-1, 5*40*55)
        x = self.dropout(self.leakyrelu(self.fc1(x)))
        x = self.dropout(self.leakyrelu(self.fc2(x)))
        x = F.log_softmax(self.fc3(x),dim=1)
        return x

