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
    """
    Backbone architecture for YOLOv3, the layer configuration is
    taken from the YOLOv3 paper
    """
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


    
class FPN(nn.Module):
    """
    Implements FPN part of YOLOv3 architecture:
    Takes outputs c3, c4, c5 from the backbone output and
    returns outputs p3, p4, p5 as described on FPN paper.
    c5          -> conv 1x1 ->      p5 
    ^                               V
    CNNBlock                        UpSample
    ^                               V
    c4          -> conv 1x1 ->      p4
    ^
    CNNBlock                        UpSample
    ^                               V
    c3          -> conv 1x1 ->      p3
    ^                              
    CNNBlock
    ^
    c2
    ^
    CNNBlock
    ^
    c1
    ^
    CNNBlock
    ^
    in
    """
    def __init__(self) -> None:
        super().__init__()

        # Upsampler, nearest essentially creates 4 copies of a value in a 2x2 grid.
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # conv_ci_pi means convolution that connects ci to pi
        # conv_sum_pi means convolution that is applied to the sum at stage i.
        self.conv_c5_p5 = CNNBlock(1024, 256, 1, 1, 0)

        self.conv_c4_p4 = CNNBlock(512, 256, 1, 1, 0)
        self.conv_sum_p4 = CNNBlock(256, 256, 1, 1, 0)

        self.conv_c3_p3 = CNNBlock(256, 256, 1, 1, 0)
        self.conv_sum_p3 = CNNBlock(256, 256, 1, 1, 0)

    def forward(self, c3, c4, c5):
        p5 = self.conv_c5_p5(c5)
        p4 = self.conv_sum_p4(self.upsample(p5) + self.conv_c4_p4(c4))
        p3 = self.conv_sum_p3(self.upsample(p4) + self.conv_c3_p3(c3))

        return p3, p4, p5
    
class DetectionHead(nn.Module):
    """
    Converts the output of FPN into detections.
    Input: corresponding stage of FPN
    Output: Vector of shape: (#batch, #boxes, G, G, (5 + #C))
    were 5 stands for 4 dimensions and objectness, #C stands for the  
    number of classes, #boxes is the number of anchor boxes for each cell.
    Finally, G is the grid resolution.
    Note: Feature map sizes for FPN outputs:
        p3: 32x32
        p4: 16x16
        p5: 8x8
    """
    def __init__(self, num_classes, num_anchors) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors
        # output length for each bounding box:
        self.box_out = 5 + self.num_classes

        # convolutional layer for each detection:
        self.conv3 = CNNBlock(256, self.num_anchors * self.box_out, 1, 1, 0)
        self.conv4 = CNNBlock(256, self.num_anchors * self.box_out, 1, 1, 0)
        self.conv5 = CNNBlock(256, self.num_anchors * self.box_out, 1, 1, 0)

    def forward(self, p3, p4, p5):

        b, _, gx, gy = p3.shape # batch, number of channels, feature map resolution
        d3 = self.conv3(p3).view(b, self.num_anchors, gx, gy, self.box_out)

        b, _, gx, gy = p4.shape
        d4 = self.conv4(p4).view(b, self.num_anchors, gx, gy, self.box_out)

        b, _, gx, gy = p5.shape
        d5 = self.conv5(p5).view(b, self.num_anchors, gx, gy, self.box_out)

        return d3, d4, d5



    
class YOLOv3(nn.Module):
    """
    YOLOv3 model put together.
    First, input image goes through Darknet53 backbone, then FPN is applied
    Finally, the 3 output detections are obtained through detection heads.
    """
    def __init__(self, num_classes = 13, num_anchors = 3) -> None:
        super().__init__()

        self.darknet = Darknet53()
        self.fpn = FPN()
        self.detection_head = DetectionHead(num_classes, num_anchors)

    def forward(self, x):
        _, _, c3, c4, c5 = self.darknet(x)

        p3, p4, p5 = self.fpn(c3, c4, c5)

        d3, d4, d5 = self.detection_head(p3, p4, p5)

        return d3, d4, d5


if __name__=="__main__":
    model = Darknet53()
    # summary(model, input_size=(16, 3, 256, 256))
    # Verifying output dimensions
    # input = torch.rand(16, 3, 256, 256)
    # output = model(input)
    # for out in output:
    #     print(out.size())
    model = YOLOv3()
    summary(model, input_size=(16, 3, 256, 256))

# torchinfo output for Full Model:
"""
===============================================================================================
Layer (type:depth-idx)                        Output Shape              Param #
===============================================================================================
YOLOv3                                        [16, 3, 32, 32, 18]       --
├─Darknet53: 1-1                              [16, 64, 128, 128]        --
│    └─Sequential: 2-1                        [16, 64, 128, 128]        --
│    │    └─CNNBlock: 3-1                     [16, 32, 256, 256]        928
│    │    └─CNNBlock: 3-2                     [16, 64, 128, 128]        18,560
│    │    └─ResBlock: 3-3                     [16, 64, 128, 128]        20,576
│    └─Sequential: 2-2                        [16, 128, 64, 64]         --
│    │    └─CNNBlock: 3-4                     [16, 128, 64, 64]         73,984
│    │    └─ResBlock: 3-5                     [16, 128, 64, 64]         82,112
│    │    └─ResBlock: 3-6                     [16, 128, 64, 64]         82,112
│    └─Sequential: 2-3                        [16, 256, 32, 32]         --
│    │    └─CNNBlock: 3-7                     [16, 256, 32, 32]         295,424
│    │    └─ResBlock: 3-8                     [16, 256, 32, 32]         328,064
│    │    └─ResBlock: 3-9                     [16, 256, 32, 32]         328,064
│    │    └─ResBlock: 3-10                    [16, 256, 32, 32]         328,064
│    │    └─ResBlock: 3-11                    [16, 256, 32, 32]         328,064
│    │    └─ResBlock: 3-12                    [16, 256, 32, 32]         328,064
│    │    └─ResBlock: 3-13                    [16, 256, 32, 32]         328,064
│    │    └─ResBlock: 3-14                    [16, 256, 32, 32]         328,064
│    │    └─ResBlock: 3-15                    [16, 256, 32, 32]         328,064
│    └─Sequential: 2-4                        [16, 512, 16, 16]         --
│    │    └─CNNBlock: 3-16                    [16, 512, 16, 16]         1,180,672
│    │    └─ResBlock: 3-17                    [16, 512, 16, 16]         1,311,488
│    │    └─ResBlock: 3-18                    [16, 512, 16, 16]         1,311,488
│    │    └─ResBlock: 3-19                    [16, 512, 16, 16]         1,311,488
│    │    └─ResBlock: 3-20                    [16, 512, 16, 16]         1,311,488
│    │    └─ResBlock: 3-21                    [16, 512, 16, 16]         1,311,488
│    │    └─ResBlock: 3-22                    [16, 512, 16, 16]         1,311,488
│    │    └─ResBlock: 3-23                    [16, 512, 16, 16]         1,311,488
│    │    └─ResBlock: 3-24                    [16, 512, 16, 16]         1,311,488
│    └─Sequential: 2-5                        [16, 1024, 8, 8]          --
│    │    └─CNNBlock: 3-25                    [16, 1024, 8, 8]          4,720,640
│    │    └─ResBlock: 3-26                    [16, 1024, 8, 8]          5,244,416
│    │    └─ResBlock: 3-27                    [16, 1024, 8, 8]          5,244,416
│    │    └─ResBlock: 3-28                    [16, 1024, 8, 8]          5,244,416
│    │    └─ResBlock: 3-29                    [16, 1024, 8, 8]          5,244,416
├─FPN: 1-2                                    [16, 256, 32, 32]         --
│    └─CNNBlock: 2-6                          [16, 256, 8, 8]           --
│    │    └─Sequential: 3-30                  [16, 256, 8, 8]           262,656
│    └─Upsample: 2-7                          [16, 256, 16, 16]         --
│    └─CNNBlock: 2-8                          [16, 256, 16, 16]         --
│    │    └─Sequential: 3-31                  [16, 256, 16, 16]         131,584
│    └─CNNBlock: 2-9                          [16, 256, 16, 16]         --
│    │    └─Sequential: 3-32                  [16, 256, 16, 16]         66,048
│    └─Upsample: 2-10                         [16, 256, 32, 32]         --
│    └─CNNBlock: 2-11                         [16, 256, 32, 32]         --
│    │    └─Sequential: 3-33                  [16, 256, 32, 32]         66,048
│    └─CNNBlock: 2-12                         [16, 256, 32, 32]         --
│    │    └─Sequential: 3-34                  [16, 256, 32, 32]         66,048
├─DetectionHead: 1-3                          [16, 3, 32, 32, 18]       --
│    └─CNNBlock: 2-13                         [16, 54, 32, 32]          --
│    │    └─Sequential: 3-35                  [16, 54, 32, 32]          13,932
│    └─CNNBlock: 2-14                         [16, 54, 16, 16]          --
│    │    └─Sequential: 3-36                  [16, 54, 16, 16]          13,932
│    └─CNNBlock: 2-15                         [16, 54, 8, 8]            --
│    │    └─Sequential: 3-37                  [16, 54, 8, 8]            13,932
===============================================================================================
Total params: 41,203,268
Trainable params: 41,203,268
Non-trainable params: 0
Total mult-adds (G): 152.20
===============================================================================================
Input size (MB): 12.58
Forward/backward pass size (MB): 2304.48
Params size (MB): 164.81
Estimated Total Size (MB): 2481.87
===============================================================================================
"""