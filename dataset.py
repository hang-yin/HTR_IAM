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
from model import *
from tokenizer import *
import h5py
from preproc import *
import pickle

'''
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

'''
full_dataset = IAMDataset()
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_set, test_set = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_dataloader = DataLoader(train_set, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_set, batch_size=1, shuffle=True)
'''
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
'''

class DataGenerator(Dataset):
    """Generator class with data streaming"""

    def __init__(self, source,charset, max_text_length, split, transform):
        self.tokenizer = Tokenizer(charset, max_text_length)
        self.transform = transform
        
        self.split = split
        self.dataset = dict()


        with open(f"{split}_signatures.pickle", 'rb') as data1:
            signatures = pickle.load(data1)
        with open(f"{split}_labels.pickle", 'rb') as data2:
            labels = pickle.load(data2)
        
        self.dataset[self.split] = dict()
        self.dataset[self.split]['dt'] = signatures
        self.dataset[self.split]['gt'] = labels

        # randomize data sequence
        randomize = np.arange(len(self.dataset[self.split]['gt']))
        np.random.seed(42)
        np.random.shuffle(randomize)

        self.dataset[self.split]['dt'] = self.dataset[self.split]['dt'][randomize]
        self.dataset[self.split]['gt'] = self.dataset[self.split]['gt'][randomize]

        # do I need to do tokenizer.encode in reader file and decode it here?
        # assumption: we need to encode, decode, and encode again because 
        # we need max_length from first pass so that we can add padding

        self.size = len(self.dataset[self.split]['gt'])
        
        '''
        with h5py.File(source, "r") as f:
            self.dataset[self.split] = dict()

            self.dataset[self.split]['dt'] = np.array(f[self.split]['dt'])
            self.dataset[self.split]['gt'] = np.array(f[self.split]['gt'])
          
            randomize = np.arange(len(self.dataset[self.split]['gt']))
            np.random.seed(42)
            np.random.shuffle(randomize)

            self.dataset[self.split]['dt'] = self.dataset[self.split]['dt'][randomize]
            self.dataset[self.split]['gt'] = self.dataset[self.split]['gt'][randomize]

            # decode sentences from byte
            self.dataset[self.split]['gt'] = [x.decode() for x in self.dataset[self.split]['gt']]
            
        self.size = len(self.dataset[self.split]['gt'])
        '''


    def __getitem__(self, i):


        signature = self.dataset[self.split]['dt'][i]
        
        #making image compatible with resnet
        # img = np.repeat(img[..., np.newaxis],3, -1)  # -1 means last axis

        # use normalization function from preproc.py file  
        # img = normalization(img)
        
        if self.transform is not None:
            signature = self.transform(signature)

        y_train = self.tokenizer.encode(self.dataset[self.split]['gt'][i]) 
        
        #padding till max length
        y_train = np.pad(y_train, (0, self.tokenizer.maxlen - len(y_train)))

        gt = torch.Tensor(y_train)

        # from here, the output size of signature is [128, width, 7]
        return signature, gt          

    def __len__(self):
      return self.size
