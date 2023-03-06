"""
Implementation of dense connections
"""
import torch
import torch.nn as nn
from torchinfo import summary
from typing import List


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size_mult) -> None:
        super().__init__()
        self.model_bn = nn.Sequential(
            nn.BatchNorm2d(in_channels),   # Batch normalization for the number of channels that comes from convolution
            nn.LeakyReLU(0.1),               # Leaky ReLU activation 
            nn.Conv2d(                      # Convolutional block with the given parameters
                in_channels=in_channels,
                out_channels=growth_rate * bn_size_mult, 
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False                  # No need for bias term since we're using batch normalization
            )
        )
        self.model_prop = nn.Sequential(
            nn.BatchNorm2d(growth_rate * bn_size_mult),   # Batch normalization for the number of channels that comes from convolution
            nn.LeakyReLU(0.1),               # Leaky ReLU activation 
            nn.Conv2d(                      # Convolutional block with the given parameters
                in_channels=growth_rate * bn_size_mult,
                out_channels=growth_rate, 
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False                  # No need for bias term since we're using batch normalization
            )
        )

    def forward(self, x):
        # Create bottleneck connection: 
        bn_output = self.model_bn(x)
        # return final output
        return self.model_prop(bn_output)

class DenseBlock(nn.Module):
    def __init__(self, growth_rate, num_layers, in_channels, bn_size_mult) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.growth_rate = growth_rate
        self.in_channels = in_channels
        self.bn_size_mult = bn_size_mult
        self.model_bank = [
            DenseLayer(in_channels + i * growth_rate, growth_rate, bn_size_mult)
            for i in range(num_layers)
        ]

    def forward(self, x):
        accumulator = [x]
        for i in range(self.num_layers):
            concat = torch.cat(accumulator, 1)
            x = self.model_bank[i](concat)
            accumulator.append(x)
        return torch.cat(accumulator, 1)


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.pool = nn.AvgPool2d(
            kernel_size=2, 
            stride=2
        )

    def forward(self, x):
        x = self.conv(x)
        return self.pool(x)
    
class DenseNet(nn.Module):
    def __init__(self, 
        growth_rate: int = 32,
        block_config = (6, 12, 24, 16),
        num_init_features: int = 64,
        bn_size_mult: int = 4,
        num_classes: int = 1000
    ) -> None:
        super().__init__()

        # First stage
        self.features = nn.Sequential(
            *[
                nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(num_init_features),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            ]
        )

        # Dense blocks:
        self.blocks = []
        in_channels = num_init_features
        for i, num_layers in enumerate(block_config):
            self.blocks.append(
                DenseBlock(growth_rate, num_layers, in_channels, bn_size_mult)
            )
            in_channels = in_channels + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = Transition(in_channels=in_channels, out_channels=in_channels // 2)
                self.blocks.append(trans)
                in_channels = in_channels // 2

        self.blocks.append(nn.BatchNorm2d(in_channels))
        self.blocks.append(nn.ReLU())

        # Last stage:
        self.final = nn.Sequential(
            nn.MaxPool2d(kernel_size=7, stride = 1, padding=0),
            nn.Flatten(),
            nn.Linear(in_channels, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        for block in self.blocks:
            print(x.size())
            x = block(x)
        print(x.size())
        x = self.final(x)

        return x

    

if __name__=="__main__":
    # Verify DenseLayer:
    # model = DenseLayer(64, 16, 2)
    # input = torch.rand(16, 64, 224, 224)
    # output = model(input)
    # print(output.size())
    # summary(model, input_size=(16, 64, 224, 224))

    # Verify DenseBlock:
    # model = DenseBlock(16, 6, 64, 4)
    # input = torch.rand(16, 64, 224, 224)
    # output = model(input)
    # print(output.size())
    # summary(model, input_size=(16, 64, 224, 224))

    # Verify DenseNet:
    """
    3 224 -- original
    64 112 -- after first layer
    64 56 -- before first block
    64 + 16*6, 56 after first block
    64 + 16*(6+12), 56 after first block
    64 + 16*(6+12+24), 56 after first block
    64 + 16*(6+12+24+16), 56 after first block
    """

    model = DenseNet(16, (6, 12, 24, 16), 64, 4)
    input = torch.rand(16, 3, 224, 224)
    output = model(input)
    print(output.size()) # torch.Size([16, 516, 28, 28])
    # summary(model, input_size=(16, 3, 224, 224))



    # model = DenseNet()
    # # summary(model, input_size=(16, 3, 256, 256))
    # # Verifying output dimensions
    # input = torch.rand(16, 3, 224, 224)
    # output = model(input)
    # for out in output:
    #     print(out.size())
    # summary(model, input_size=(16, 3, 224, 224))