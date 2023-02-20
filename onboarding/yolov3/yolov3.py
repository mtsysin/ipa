"""
Implementation of YOLYv1 paper for the BDD100k dataset, mainly getting familiar with the workflow and
trying to get familiar with what the previous semester's team did.
Main model file
"""
import torch
import torch.nn as nn
from torchinfo import summary
from typing import List


class ResBlock(nn.Module):
    """
    Residual block used in Darknet-53 architecture
    Input: list of dict parameters for each convolutional block
    """
    def __init__(self, *conv_blocks_data: List[List]) -> None:
        super().__init__()

        conv_blocks = []

        for conv_block_data in conv_blocks_data:
            conv_blocks.append(
                nn.Conv2d(
                    *conv_block_data
                )
            )

        self.model = nn.Sequential(*conv_blocks)
    
    def forward(self, x):
        return self.model(x) + x


class CNNBlock(nn.Module):
    """
    Create a convolutional block which repeatedly appears in the architecture.
    Takes an image of size (batch, in_channels, x, y)
    Outputs (batch, out_channels, x//stride, y//stride)
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
    

class Darknet53(nn.Module):
    """Backbone architecture for YOLOv3"""
    def __init__(self) -> None:
        super().__init__()
        self.phase1 = nn.Sequential(
            CNNBlock(3, 32, 3, 1, 1),
            CNNBlock(32, 64, 3, 2, 1),
            ResBlock([64, 32, 1, 1, 0], [32, 64, 3, 1, 1])
        )
        self.phase2 = nn.Sequential(
            CNNBlock(64, 128, 3, 2, 1),
            *[ResBlock([128, 64, 1, 1, 0], [64, 128, 3, 1, 1]) for _ in range(2)]
        )
        self.phase3 = nn.Sequential(
            CNNBlock(128, 256, 3, 2, 1),
            *[ResBlock([256, 128, 1, 1, 0], [128, 256, 3, 1, 1]) for _ in range(8)]
        )
        self.phase4 = nn.Sequential(
            CNNBlock(256, 512, 3, 2, 1),
            *[ResBlock([512, 256, 1, 1, 0], [256, 512, 3, 1, 1]) for _ in range(8)]
        )
        self.phase5 = nn.Sequential(
            CNNBlock(512, 1024, 3, 2, 1),
            *[ResBlock([1024, 512, 1, 1, 0], [512, 1024, 3, 1, 1]) for _ in range(4)]
        )

    def forward(self, x):
        c1 = self.phase1(x)
        c2 = self.phase2(c1)
        c3 = self.phase3(c2)
        c4 = self.phase4(c3)
        c5 = self.phase5(c4)

        return c1, c2, c3, c4, c5


if __name__=="__main__":
    model = Darknet53()
    summary(model, input_size=(16, 3, 256, 256))
    # Verifying output dimensions
    input = torch.rand(16, 3, 256, 256)
    output = model(input)
    for out in output:
        print(out.size())

# torchinfo output for Darknet:
"""==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
Darknet53                                [16, 64, 128, 128]        --
├─Sequential: 1-1                        [16, 64, 128, 128]        --
│    └─CNNBlock: 2-1                     [16, 32, 256, 256]        --
│    │    └─Sequential: 3-1              [16, 32, 256, 256]        928
│    └─CNNBlock: 2-2                     [16, 64, 128, 128]        --
│    │    └─Sequential: 3-2              [16, 64, 128, 128]        18,560
│    └─ResBlock: 2-3                     [16, 64, 128, 128]        --
│    │    └─Sequential: 3-3              [16, 64, 128, 128]        20,576
├─Sequential: 1-2                        [16, 128, 64, 64]         --
│    └─CNNBlock: 2-4                     [16, 128, 64, 64]         --
│    │    └─Sequential: 3-4              [16, 128, 64, 64]         73,984
│    └─ResBlock: 2-5                     [16, 128, 64, 64]         --
│    │    └─Sequential: 3-5              [16, 128, 64, 64]         82,112
│    └─ResBlock: 2-6                     [16, 128, 64, 64]         --
│    │    └─Sequential: 3-6              [16, 128, 64, 64]         82,112
├─Sequential: 1-3                        [16, 256, 32, 32]         --
│    └─CNNBlock: 2-7                     [16, 256, 32, 32]         --
│    │    └─Sequential: 3-7              [16, 256, 32, 32]         295,424
│    └─ResBlock: 2-8                     [16, 256, 32, 32]         --
│    │    └─Sequential: 3-8              [16, 256, 32, 32]         328,064
│    └─ResBlock: 2-9                     [16, 256, 32, 32]         --
│    │    └─Sequential: 3-9              [16, 256, 32, 32]         328,064
│    └─ResBlock: 2-10                    [16, 256, 32, 32]         --
│    │    └─Sequential: 3-10             [16, 256, 32, 32]         328,064
│    └─ResBlock: 2-11                    [16, 256, 32, 32]         --
│    │    └─Sequential: 3-11             [16, 256, 32, 32]         328,064
│    └─ResBlock: 2-12                    [16, 256, 32, 32]         --
│    │    └─Sequential: 3-12             [16, 256, 32, 32]         328,064
│    └─ResBlock: 2-13                    [16, 256, 32, 32]         --
│    │    └─Sequential: 3-13             [16, 256, 32, 32]         328,064
│    └─ResBlock: 2-14                    [16, 256, 32, 32]         --
│    │    └─Sequential: 3-14             [16, 256, 32, 32]         328,064
│    └─ResBlock: 2-15                    [16, 256, 32, 32]         --
│    │    └─Sequential: 3-15             [16, 256, 32, 32]         328,064
├─Sequential: 1-4                        [16, 512, 16, 16]         --
│    └─CNNBlock: 2-16                    [16, 512, 16, 16]         --
│    │    └─Sequential: 3-16             [16, 512, 16, 16]         1,180,672
│    └─ResBlock: 2-17                    [16, 512, 16, 16]         --
│    │    └─Sequential: 3-17             [16, 512, 16, 16]         1,311,488
│    └─ResBlock: 2-18                    [16, 512, 16, 16]         --
│    │    └─Sequential: 3-18             [16, 512, 16, 16]         1,311,488
│    └─ResBlock: 2-19                    [16, 512, 16, 16]         --
│    │    └─Sequential: 3-19             [16, 512, 16, 16]         1,311,488
│    └─ResBlock: 2-20                    [16, 512, 16, 16]         --
│    │    └─Sequential: 3-20             [16, 512, 16, 16]         1,311,488
│    └─ResBlock: 2-21                    [16, 512, 16, 16]         --
│    │    └─Sequential: 3-21             [16, 512, 16, 16]         1,311,488
│    └─ResBlock: 2-22                    [16, 512, 16, 16]         --
│    │    └─Sequential: 3-22             [16, 512, 16, 16]         1,311,488
│    └─ResBlock: 2-23                    [16, 512, 16, 16]         --
│    │    └─Sequential: 3-23             [16, 512, 16, 16]         1,311,488
│    └─ResBlock: 2-24                    [16, 512, 16, 16]         --
│    │    └─Sequential: 3-24             [16, 512, 16, 16]         1,311,488
├─Sequential: 1-5                        [16, 1024, 8, 8]          --
│    └─CNNBlock: 2-25                    [16, 1024, 8, 8]          --
│    │    └─Sequential: 3-25             [16, 1024, 8, 8]          4,720,640
│    └─ResBlock: 2-26                    [16, 1024, 8, 8]          --
│    │    └─Sequential: 3-26             [16, 1024, 8, 8]          5,244,416
│    └─ResBlock: 2-27                    [16, 1024, 8, 8]          --
│    │    └─Sequential: 3-27             [16, 1024, 8, 8]          5,244,416
│    └─ResBlock: 2-28                    [16, 1024, 8, 8]          --
│    │    └─Sequential: 3-28             [16, 1024, 8, 8]          5,244,416
│    └─ResBlock: 2-29                    [16, 1024, 8, 8]          --
│    │    └─Sequential: 3-29             [16, 1024, 8, 8]          5,244,416
==========================================================================================
Total params: 40,569,088
Trainable params: 40,569,088
Non-trainable params: 0
Total mult-adds (G): 148.68
==========================================================================================
Input size (MB): 12.58
Forward/backward pass size (MB): 2113.93
Params size (MB): 162.28
Estimated Total Size (MB): 2288.79
==========================================================================================
torch.Size([16, 64, 128, 128])
torch.Size([16, 128, 64, 64])
torch.Size([16, 256, 32, 32])
torch.Size([16, 512, 16, 16])
torch.Size([16, 1024, 8, 8])"""