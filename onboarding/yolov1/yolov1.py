"""
Implementation of YOLYv1 paper for the BDD100k dataset, mainly getting familiar with the workflow and
trying to get familiar with what the previous semester's team did.
"""
import torch
import torch.nn as nn
from torchinfo import summary


class CNNBlock(nn.Module):
    """
    Create a convolutional block which repeatedly appears in the architecture.
    Takes an image of size (in_channels, )
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(                      # Convolutional block with the given parameters
                in_channels=in_channels,
                out_channels=out_channels, 
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False                  # No need for bias term since we're using batch normalization
            ),
            nn.BatchNorm2d(out_channels),   # Batch normalization for the number of channels that comes from convolution
            nn.LeakyReLU(0.1)               # Leaky ReLU activation 
        )

    def forward(self, x):
        return self.model(x)

class YOLOv1(nn.Module):
    def __init__(self, in_channels=3, grids=7, num_bboxes=2, num_classes=13) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.S = grids
        self.B = num_bboxes
        self.C = num_classes
        # generate colvolutional blocks.
        self.convs = self._conv_layers()
        self.fcn = self._fc_layers()

    def forward(self, x):
        return self.fcn(self.convs(x)).view(-1, self.S, self.S, (self.B * 5 + self.C))
        # return self.convs(x)


    def _conv_layers(self):
        return nn.Sequential(
            CNNBlock(
                in_channels=3,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3
            ),                                          # (64, 224, 224)
            nn.MaxPool2d(kernel_size=2, stride=2),      # (64, 112, 112)

            CNNBlock(64, 192, 3, 1, 1),                 # (192, 112, 112)
            nn.MaxPool2d(2, 2),                         # (192, 56, 56)

            CNNBlock(192, 128, 1, 1, 0),                # (128, 56, 56)
            CNNBlock(128, 256, 3, 1, 1),                # (256, 56, 56)
            CNNBlock(256, 256, 1, 1, 0),                # (256, 56, 56)
            CNNBlock(256, 512, 3, 1, 1),                # (512, 56, 56)
            nn.MaxPool2d(2, 2),                         # (512, 28, 28)

            # 4 identical blocks:
            *sum([
                [CNNBlock(512, 256, 1, 1, 0), CNNBlock(256, 512, 3, 1, 1)] for _ in range(4)
            ], []),
            CNNBlock(512, 512, 1, 1, 0),                # (512, 28, 28)
            CNNBlock(512, 1024, 3, 1, 1),               # (1024, 28, 28)
            nn.MaxPool2d(2, 2),                         # (1024, 14, 14)

            # 2 identical blocks:
            *sum([
                [CNNBlock(1024, 512, 1, 1, 0), CNNBlock(512, 1024, 3, 1, 1)] for _ in range(2)
            ], []),
            CNNBlock(1024, 1024, 3, 1, 1),              # (1024, 14, 14)
            CNNBlock(1024, 1024, 3, 2, 1),              # (1024, 7, 7)

            CNNBlock(1024, 1024, 3, 1, 1),              # (1024, 7, 7)
            CNNBlock(1024, 1024, 3, 1, 1),              # (1024, 7, 7)
        )

    def _fc_layers(self):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024*7*7, 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, self.S * self.S * (self.B * 5 + self.C))
        )

if __name__=="__main__":
    model = YOLOv1()
    summary(model, input_size=(16, 3, 448, 448))


