import torch
from torch import nn
import numpy as np

class CNTRANSFORMER(nn.Module):
    def __init__(self):
        super().__init__()
        #self.constructor
        self.cnn_extractor = nn.Sequential(
            nn.Conv2d(7, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 
            nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2)),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))

    def forward(self, my_input):
        my_input = self.cnn_extractor(my_input.float())
        print(my_input.shape)
        
        
    