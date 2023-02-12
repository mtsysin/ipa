"""
Implementation of YOLYv1 paper for the BDD100k dataset, mainly getting familiar with the workflow and
trying to get familiar with what the previous semester's team did.
Loss function
"""
import torch
import torch.nn as nn
from torchinfo import summary
from helpers import iou
import numpy as np
import random

EPS = 1e-6

class YOLOLoss(nn.Module):
    """
    Loss for YOLO network, implemented based on the original paper.
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

    def forward(self, pred, target):
        """
        The function takes a prediction tensor of shape (batch_size, S, S, 5*B + C)
        and target tensor of shape (batch_size, S, S, B + C)
        The last dimension is organized as [class probabilities, (score, x,y,w,h)]
        The output is a single float number -- resulting loss
        """

        # print(pred[3, 5, 6, ...], target[3, 5, 6, ...])

        # We will compute all parts of loss function as described in the paper:

        # We need a class which will calculate MSE losses:
        mse = nn.MSELoss(reduction='sum')
        # Then, we need to figure out which bounding box is associated with an object:
        # Get IOU scores for each bounding box. The resulting tensor is of shape (batch_size, S, S, B)
        iou_pred = torch.cat(
            [iou(pred[..., self.C + 1 + i * 5: self.C + 5 + i * 5], target[..., self.C + 1: self.C + 5]) for i in range(self.B)],
            dim=-1
        )

        # Find maximum for each cell
        _, bbox_ind = torch.max(iou_pred, dim = -1)
        # Convert to correct shape
        bbox_ind = torch.unsqueeze(bbox_ind, dim = -1)

        # Find grids that contain objects
        Iobj_i = target[..., self.C].unsqueeze(dim = -1)

        # Now, continue with the rest of the loss:
        loss = 0
        # 1. Loss associated with localization
        # Create mask that tells which indices to load
        bbox_resp_mask = torch.cat(
            [self.C + 1 + bbox_ind * 5 + i for i in range(4)],
            dim=-1
        )
        # Get bounding box targets and predictions
        bbox_pred = Iobj_i * torch.gather(pred, dim=-1, index=bbox_resp_mask)
        # VERSION WITHOUT GATHER:
        # bbox_pred = Iobj_i * sum([ * pred[..., self.C + 1 + 5 * b] for b in self.B])

        bbox_target = Iobj_i * target[..., self.C + 1 : self.C + 5]

        # Take a square root of width and height
        bbox_pred[..., 2:4] = torch.sign(bbox_pred[...,2:4]) * torch.sqrt(torch.abs(bbox_pred[..., 2:4] + EPS)) 
        bbox_target[..., 2:4] = torch.sqrt(bbox_target[..., 2:4])

        loss += self.lambda_coord * mse(bbox_pred, bbox_target)

        # 2. Loss associated with classification
        loss += mse(Iobj_i * pred[..., :self.C], Iobj_i * target[..., :self.C])

        # 3. Loss associated with confidence 
        conf_pred = torch.gather(pred, dim=-1, index = self.C + 5 * bbox_ind)
        conf_target = target[..., self.C].unsqueeze(-1)
        # 3.1 Object is present
        loss += mse(Iobj_i * conf_pred, Iobj_i * conf_target)
        # 3.2 Object is absent
        # If object is absent no bounding box is responsible for finding the object, thus we take loss for all confidence values
        loss += self.lambda_noobj * mse((1 - Iobj_i) * pred[..., self.C::5], (1 - Iobj_i) * conf_target.expand(-1, -1, -1, 2))

        # VERSION WITHOUT GATHER:


        return loss

if __name__=="__main__":
    pred = torch.rand(16, 7, 7, 23)
    target = torch.rand(16, 7, 7, 23)
    loss = YOLOLoss()
    seed= 100
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    preds = torch.rand(16, 7, 7, 23)
    target = torch.rand(16, 7, 7, 18)
    print(loss(preds, target))
    # output = iou(pred, target)
    # print(output.size())