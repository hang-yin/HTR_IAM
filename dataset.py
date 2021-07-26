import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from os import listdir
from os.path import isfile, join
import sys
import numpy as np
from PSFforIAM import *

class IAMDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform = None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label