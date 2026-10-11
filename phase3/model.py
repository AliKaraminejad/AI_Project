import torch
from torch import nn


class SignLanguageCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.nnl1 = nn.Linear(128 * 8 * 8, 512)
        self.dropout = nn.Dropout(0.5)
        self.nnl2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.pool(self.relu(self.bn1(self.conv1(inputs))))
        inputs = self.pool(self.relu(self.bn2(self.conv2(inputs))))
        inputs = self.pool(self.relu(self.bn3(self.conv3(inputs))))
        inputs = torch.flatten(inputs, 1)
        inputs = self.relu(self.nnl1(inputs))
        inputs = self.dropout(inputs)
        return self.nnl2(inputs)
