"""network3.py
~~~~~~~~~~~~~~
Class to work with datasets
"""

#### Libraries
# Standard library
import pickle
import gzip
import os
import time
import threading
# Third-party libraries
import numpy as np
import glob
import timeit
import sys
import csv
from utils import normalize_3D_box, maps_to_submaps_4D, maps_to_submaps
from utils import element_dict
import protein_atoms
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except ImportError:
  print("run without TORCH")
import re
from utils import Atom
import utils

ATOM_NAMES=["C","S","H","N","O"]
VOX_SIZE = 1.0
RESOLUTION = 3.0
VX_BOX_SIZE = 15
MAP_BOX_SIZE = 11
N_SAMPLS_FOR_1V3 = 1.0/(2.0**3)
N_CHANNELS = 5


MEAN_MAX = 1
MEAN_MIN = 0.05
MEAN = (MEAN_MAX+MEAN_MIN)/2

STD_MIN = 0.05
STD_MAX = 0.3
SIGMA = (STD_MIN+STD_MAX)/2

L_NORM = 50

BATCH_SIZE = 128
N_LABELS = 22

AA_BoxIn = 11
AA_BoxOut = 1
AA_HALF_BOX = (AA_BoxIn-AA_BoxOut)//2
AA_LIN =AA_BoxIn*4
AA_LOUT=AA_LIN - (AA_BoxIn-AA_BoxOut)



class PATCHES_DATASET(Dataset):

    def __init__(self, db_name, list_file):
        """
        Args:
         TBD
        """
        self.name = db_name
        self.files_list = []

        with open(list_file,'r') as f:
            patch_file = f.readline()
            while patch_file != "":
                self.files_list.append(patch_file[:-1])
                patch_file = f.readline()

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        D = AA_HALF_BOX
        label_file = self.files_list[idx]
        map_file = label_file[:-10]+"_map.npy"
        inp_map = np.load(map_file)
        normalized_map  = normalize_3D_box(inp_map, mean=MEAN, sigma = SIGMA) 
        em_map = np.expand_dims(normalized_map,0)
        labels_map = np.load(label_file)
        un_norm_map = inp_map[D:-D,D:-D,D:-D]

        loc_string = re.findall(r'X\d+_Y\d+_Z\d+',map_file)[0]

        x = int(re.findall(r'X\d+',loc_string)[0].strip('X'))
        y = int(re.findall(r'Y\d+',loc_string)[0].strip('Y'))
        z = int(re.findall(r'Z\d+',loc_string)[0].strip('Z'))

        #create atoms
        sample ={"em_map": em_map, "label":labels_map, \
            "file_name": label_file, "un_norm_map":un_norm_map, "corner_pos": (x,y,z)}
        return sample

