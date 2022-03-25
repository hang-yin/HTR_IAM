import os
import html
import string
import multiprocessing
import xml.etree.ElementTree as ET

from glob import glob
from tqdm import tqdm
from data import preproc as pp
from functools import partial
from PSFforIAM import *

FILE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(FILE_PATH, "data/")
LABEL_DATA_PATH = os.path.join(DATA_PATH, "ascii/")
STROKES_DATA_PATH = os.path.join(DATA_PATH, "lineStrokes/")

ESCAPE_CHAR = '~!@#$%^&*()_+{}:"<>?`-=[];\',./|\n'


class Dataset():
    """Dataset class to read images and sentences from base (raw files)"""

    def __init__(self, source, name):
        self.source = source
        self.name = name
        self.dataset = None
        self.partitions = ['train', 'valid', 'test']

    def read_partitions(self):
        """Read images and sentences from dataset"""

        dataset = getattr(self, f"_{self.name}")()

        if not self.dataset:
            self.dataset = dict()

            for y in self.partitions:
                self.dataset[y] = {'dt': [], 'gt': []}

        for y in self.partitions:
            self.dataset[y]['dt'] += dataset[y]['dt']
            self.dataset[y]['gt'] += dataset[y]['gt']

    def _iam(self):
        """IAM dataset reader"""

        # define paths to train, validation, and test set info
        pt_path = os.path.join(self.source, "partition_info")
        paths = {"train": open(os.path.join(pt_path, "trainsetsub.txt")).read().splitlines(),
                 "valid": open(os.path.join(pt_path, "validation1.txt")).read().splitlines(),
                 "test": open(os.path.join(pt_path, "test.txt")).read().splitlines()}
        
        # initialize dataset as an empty dictionary
        dataset = dict()

        # loop through all three partitions
        for i in self.partitions: 
            # initialize one dataset entry, including data and groundtruth
            dataset[i] = {"dt": [], "gt": []}

            

        lines = open(os.path.join(self.source, "ascii", "lines.txt")).read().splitlines()
        gt_dict = dict()

        for line in lines:
            if (not line or line[0] == "#"):
                continue

            split = line.split()

            if split[1] == "ok":
                gt_dict[split[0]] = " ".join(split[8::]).replace("|", " ")

        dataset = dict()

        for i in self.partitions:
            dataset[i] = {"dt": [], "gt": []}
            # use extraction function from PSFforIAM file to extract signatures and labels
            signatures, labels = extraction(paths[i],"IAM")
            dataset[i]['dt'] = signatures
            dataset[i]['gt'] = labels

            '''
            for line in paths[i]:
                try:
                    split = line.split("-")
                    folder = f"{split[0]}-{split[1]}"

                    img_file = f"{split[0]}-{split[1]}-{split[2]}.png"
                    img_path = os.path.join(self.source, "lines", split[0], folder, img_file)

                    dataset[i]['gt'].append(gt_dict[line])
                    dataset[i]['dt'].append(img_path)
                except KeyError:
                    pass
            '''

        return dataset

    '''
    def preprocess_partitions(self, input_size):
        #Preprocess images and sentences from partitions

        for y in self.partitions:
            arange = range(len(self.dataset[y]['gt']))

            for i in reversed(arange):
                text = pp.text_standardize(self.dataset[y]['gt'][i])

                if not self.check_text(text):
                    self.dataset[y]['gt'].pop(i)
                    self.dataset[y]['dt'].pop(i)
                    continue

                self.dataset[y]['gt'][i] = text.encode()

            results = []
            with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
                print(f"Partition: {y}")
                for result in tqdm(pool.imap(partial(pp.preprocess, input_size=input_size), self.dataset[y]['dt']),
                                   total=len(self.dataset[y]['dt'])):
                    results.append(result)
                pool.close()
                pool.join()

            self.dataset[y]['dt'] = results
    '''
    
    

    '''
    @staticmethod
    def check_text(text):
        """Make sure text has more characters instead of punctuation marks"""

        strip_punc = text.strip(string.punctuation).strip()
        no_punc = text.translate(str.maketrans("", "", string.punctuation)).strip()

        if len(text) == 0 or len(strip_punc) == 0 or len(no_punc) == 0:
            return False

        punc_percent = (len(strip_punc) - len(no_punc)) / len(strip_punc)

        return len(no_punc) > 2 and punc_percent <= 0.1
    '''
