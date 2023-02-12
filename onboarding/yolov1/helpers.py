"""Helper functions for YOLOv1 implementation"""
import torch

EPSILON = 1e-6

def iou(pred, target):
    """
    Takes two tensors of length 4 where one is target and another is prediction
    The tensors are organized as (x_center, y_center, w, h)
    Shape: (batch, 4)
    """
    # Finding the intersection rectange(if exists)
    x1 = torch.max(pred[...,0] - pred[...,2] / 2, target[...,0] - target[...,2] / 2) # left boundary
    y1 = torch.max(pred[...,1] - pred[...,3] / 2, target[...,1] - target[...,3] / 2) # bottom boundary
    x2 = torch.min(pred[...,0] + pred[...,2] / 2, target[...,0] + target[...,2] / 2) # right boundary
    y2 = torch.min(pred[...,1] + pred[...,3] / 2, target[...,1] + target[...,3] / 2) # top boundary

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0) # Get the area if the resulting points are in correct order.

    # Union is equal to the sum of 2 boxes' area minus intersection
    return torch.unsqueeze(intersection / (pred[...,2] * pred[...,3] + target[...,2] * target[...,3] - intersection + EPSILON), dim = -1)

if __name__=="__main__":
    # Testing code
    pred = torch.rand(16, 4)
    target = torch.rand(16, 4)
    output = iou(pred, target)
    print(output)
    print(output.size())
    pred = torch.rand(16, 7, 7, 4)
    target = torch.rand(16, 7, 7, 4)
    output = iou(pred, target)
    print(output)
    print(output.size())


