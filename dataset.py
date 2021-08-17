import os
import torch
from torchvision.io import read_image
from os import listdir
from os.path import isfile, join
import sys
import numpy as np
from PSFforIAM import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dcback import *

FILE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(FILE_PATH, "data/")
LABEL_DATA_PATH = os.path.join(DATA_PATH, "ascii/")
STROKES_DATA_PATH = os.path.join(DATA_PATH, "lineStrokes/")

class IAMDataset(Dataset):
    def __init__(self, signature_file, label_file):
        with open(signature_file, 'rb') as f:
            self.signatures = pickle.load(f)
        with open(label_file, 'rb') as f:
            self.labels = pickle.load(f)
        #self.signatures, self.labels = extraction("data/IAM/lineStrokes",'IAM')

    def __len__(self):
        return len(self.signatures)

    def __getitem__(self, idx):
        signature = np.array(self.signatures[idx])
        signature = np.transpose(signature, (2,0,1))
        label = self.labels[idx]
        return signature, label


sub_dataset = IAMDataset('train_signatures.pickle', 'train_labels.pickle')
sub_dataloader  = DataLoader(sub_dataset, batch_size = 1, shuffle = True)

'''
full_dataset = IAMDataset()
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_set, test_set = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_dataloader = DataLoader(train_set, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_set, batch_size=1, shuffle=True)
'''

model = CNTRANSFORMER()
for train_images, train_labels in sub_dataloader:
    sample_image = train_images[0]
    sample_label = train_labels[0]
    print(train_images.shape)
    print(sample_image.shape)
    #train_images.float()
    model.forward(train_images)
    break
