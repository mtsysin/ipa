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

# torchinfo output:
"""==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
YOLOv1                                   [16, 7, 7, 23]            --
├─Sequential: 1-1                        [16, 1024, 7, 7]          --
│    └─CNNBlock: 2-1                     [16, 64, 224, 224]        --
│    │    └─Sequential: 3-1              [16, 64, 224, 224]        9,536
│    └─MaxPool2d: 2-2                    [16, 64, 112, 112]        --
│    └─CNNBlock: 2-3                     [16, 192, 112, 112]       --
│    │    └─Sequential: 3-2              [16, 192, 112, 112]       110,976
│    └─MaxPool2d: 2-4                    [16, 192, 56, 56]         --
│    └─CNNBlock: 2-5                     [16, 128, 56, 56]         --
│    │    └─Sequential: 3-3              [16, 128, 56, 56]         24,832
│    └─CNNBlock: 2-6                     [16, 256, 56, 56]         --
│    │    └─Sequential: 3-4              [16, 256, 56, 56]         295,424
│    └─CNNBlock: 2-7                     [16, 256, 56, 56]         --
│    │    └─Sequential: 3-5              [16, 256, 56, 56]         66,048
│    └─CNNBlock: 2-8                     [16, 512, 56, 56]         --
│    │    └─Sequential: 3-6              [16, 512, 56, 56]         1,180,672
│    └─MaxPool2d: 2-9                    [16, 512, 28, 28]         --
│    └─CNNBlock: 2-10                    [16, 256, 28, 28]         --
│    │    └─Sequential: 3-7              [16, 256, 28, 28]         131,584
│    └─CNNBlock: 2-11                    [16, 512, 28, 28]         --
│    │    └─Sequential: 3-8              [16, 512, 28, 28]         1,180,672
│    └─CNNBlock: 2-12                    [16, 256, 28, 28]         --
│    │    └─Sequential: 3-9              [16, 256, 28, 28]         131,584
│    └─CNNBlock: 2-13                    [16, 512, 28, 28]         --
│    │    └─Sequential: 3-10             [16, 512, 28, 28]         1,180,672
│    └─CNNBlock: 2-14                    [16, 256, 28, 28]         --
│    │    └─Sequential: 3-11             [16, 256, 28, 28]         131,584
│    └─CNNBlock: 2-15                    [16, 512, 28, 28]         --
│    │    └─Sequential: 3-12             [16, 512, 28, 28]         1,180,672
│    └─CNNBlock: 2-16                    [16, 256, 28, 28]         --
│    │    └─Sequential: 3-13             [16, 256, 28, 28]         131,584
│    └─CNNBlock: 2-17                    [16, 512, 28, 28]         --
│    │    └─Sequential: 3-14             [16, 512, 28, 28]         1,180,672
│    └─CNNBlock: 2-18                    [16, 512, 28, 28]         --
│    │    └─Sequential: 3-15             [16, 512, 28, 28]         263,168
│    └─CNNBlock: 2-19                    [16, 1024, 28, 28]        --
│    │    └─Sequential: 3-16             [16, 1024, 28, 28]        4,720,640
│    └─MaxPool2d: 2-20                   [16, 1024, 14, 14]        --
│    └─CNNBlock: 2-21                    [16, 512, 14, 14]         --
│    │    └─Sequential: 3-17             [16, 512, 14, 14]         525,312
│    └─CNNBlock: 2-22                    [16, 1024, 14, 14]        --
│    │    └─Sequential: 3-18             [16, 1024, 14, 14]        4,720,640
│    └─CNNBlock: 2-23                    [16, 512, 14, 14]         --
│    │    └─Sequential: 3-19             [16, 512, 14, 14]         525,312
│    └─CNNBlock: 2-24                    [16, 1024, 14, 14]        --
│    │    └─Sequential: 3-20             [16, 1024, 14, 14]        4,720,640
│    └─CNNBlock: 2-25                    [16, 1024, 14, 14]        --
│    │    └─Sequential: 3-21             [16, 1024, 14, 14]        9,439,232
│    └─CNNBlock: 2-26                    [16, 1024, 7, 7]          --
│    │    └─Sequential: 3-22             [16, 1024, 7, 7]          9,439,232
│    └─CNNBlock: 2-27                    [16, 1024, 7, 7]          --
│    │    └─Sequential: 3-23             [16, 1024, 7, 7]          9,439,232
│    └─CNNBlock: 2-28                    [16, 1024, 7, 7]          --
│    │    └─Sequential: 3-24             [16, 1024, 7, 7]          9,439,232
├─Sequential: 1-2                        [16, 1127]                --
│    └─Flatten: 2-29                     [16, 50176]               --
│    └─Linear: 2-30                      [16, 4096]                205,524,992
│    └─Dropout: 2-31                     [16, 4096]                --
│    └─LeakyReLU: 2-32                   [16, 4096]                --
│    └─Linear: 2-33                      [16, 1127]                4,617,319
==========================================================================================
Total params: 270,311,463
Trainable params: 270,311,463
Non-trainable params: 0
Total mult-adds (G): 324.54
==========================================================================================
Input size (MB): 38.54
Forward/backward pass size (MB): 3533.06
Params size (MB): 1081.25
Estimated Total Size (MB): 4652.84
=========================================================================================="""
