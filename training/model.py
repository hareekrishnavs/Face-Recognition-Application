import torch
import torch.nn as nn

from utils.config import dropoutRate, hiddenDim


class FaceClassifierCNN(nn.Module):
    def __init__(self, numClasses: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self._conv_block(3, 32),
            nn.MaxPool2d(2),
            self._conv_block(32, 64),
            nn.MaxPool2d(2),
            self._conv_block(64, 128),
            nn.MaxPool2d(2),
            self._conv_block(128, 192),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropoutRate),
            nn.Linear(192, hiddenDim),
            nn.BatchNorm1d(hiddenDim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropoutRate * 0.5),
            nn.Linear(hiddenDim, numClasses),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    def _conv_block(self, inChannels: int, outChannels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(inChannels, outChannels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(outChannels),
            nn.ReLU(inplace=True),
            nn.Conv2d(outChannels, outChannels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(outChannels),
            nn.ReLU(inplace=True),
        )
