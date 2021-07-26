import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from os import listdir
from os.path import isfile, join
import sys
import numpy as np
from PSFforIAM import *

FILE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(FILE_PATH, "data/")
LABEL_DATA_PATH = os.path.join(DATA_PATH, "ascii/")
STROKES_DATA_PATH = os.path.join(DATA_PATH, "lineStrokes/")

class IAMDataset(Dataset):
    def __init__(self):
        self.signatures, self.labels = extraction("data/IAM/lineStrokes",'IAM')

    def __len__(self):
        return np.size(self.signatures)[0]

    def __getitem__(self, idx):
        image = self.signatures[idx]
        label = self.labels[idx]
        return image, label