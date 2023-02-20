"""Implementation of BDD100k dataset parsing"""
import torch
from torch.utils import data

class BDD100k(data.Dataset):
    def __init__(self, path = "./bdd100k") -> None:
        super().__init__()


