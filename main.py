from pathlib import Path
import torch
import numpy as np
import argparse
import cv2
import h5py
import os
import string
import torchvision.transforms as T
from preproc import *
from tokenizer import *
from dataset import *
from model import *
from pos_encoding import *
from PSFforIAM import *
from reader import *

def main():
    #define path to data files
    FILE_PATH = os.path.dirname(os.path.realpath(__file__))
    DATA_PATH = os.path.join(FILE_PATH, "data/")
    LABEL_DATA_PATH = os.path.join(DATA_PATH, "ascii/")
    STROKES_DATA_PATH = os.path.join(DATA_PATH, "lineStrokes/")

    ESCAPE_CHAR = '~!@#$%^&*()_+{}:"<>?`-=[];\',./|\n'

    source_path = os.path.join(".", "data", "iam.hdf5")

    #extract PSF features and store them in a HDF5 file
    print("Extracting PSF features.")

    ds = Dataset(source=DATA_PATH, name="iam")
    ds.read_partitions()

    '''
    print("Partitions will be preprocessed...")
    ds.preprocess_partitions(input_size=input_size)
    '''

    print("Partitions will be saved...")
    os.makedirs(os.path.dirname(source_path), exist_ok=True)

    for i in ds.partitions:
        with h5py.File(source_path, "a") as hf:
            hf.create_dataset(f"{i}/dt", data=ds.dataset[i]['dt'], compression="gzip", compression_opts=9)
            hf.create_dataset(f"{i}/gt", data=ds.dataset[i]['gt'], compression="gzip", compression_opts=9)
            print(f"[OK] {i} partition.")

    print("PSF feature extraction completed.")


if __name__ == "__main__":
    main()