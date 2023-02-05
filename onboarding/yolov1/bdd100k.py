import torch
import torch.nn as nn
from dataclasses import dataclass

# Configuration from the YOLOv1 paper:
# Convolution layer ()
# Create NN structure parser:

@dataclass
class ConvConfig():
    out_ch: int
    size: int
    stride: int
    padding: int

print(ConvConfig(1, 2,3 ,4))

conv_config = [
    (64, 7, 2, 3),
    "Maxpool",
    (192, 3, 1, 1),
    "Maxpool",
    (128, 1, 1, 0),
    (256, 3, 1, 1),
    (256, 1, 1, 0),
    (512, 3, 1, 1),
    "Maxpool", 
    [(256, 1, 1, 0), (512, 3, 1, 1), 4],
    (512, 1, 1, 0), 
    (1024, 3, 1, 1),
    "Maxpool",
    [(512, 1, 1, 0), (1024, 3, 1, 1), 2],
    (1024, 3, 1, 1),
    (1024, 3, 2, 1),
    (1024, 3, 1, 1),
    (1024, 3, 1, 1),
]