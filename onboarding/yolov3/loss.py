"""
Implementation of YOLYv3 paper for the BDD100k dataset, mainly getting familiar with the workflow and
trying to get familiar with what the previous semester's team did.
Loss function
"""
import torch
import torch.nn as nn
from torchinfo import summary
# from helpers import iou
import numpy as np
import random

EPS = 1e-6

class YOLOv3_loss(nn.Module):
    """
    Loss for YOLOv3 network, implemented based on the original paper.
    Input parameters:
        split_grids (int): number of cells in each direction (S)
        num_bboxes (int): number of bounding boxes to predict per cell
        num_classes (int): number of dataset classes
        lambda_coord (int): weight for location loss
        lambda_noobj (int): weight for the case where there is no object in the cell.
    """
    def __init__(self, split_grids=7, num_bboxes=2, num_classes=13, lambda_coord=5, lambda_noobj=0.5) -> None:
        super().__init__()
        self.S = split_grids
        self.B = num_bboxes
        self.C = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self):
        pass
