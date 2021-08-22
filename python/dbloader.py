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

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except ImportError:
  print("run without TORCH")



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



def maps_to_submaps(inp_map, inp_label, Lin=None, Lout=None):
    Nx,Ny,Nz = inp_map.shape
    D = (Lin-Lout)//2
    x = [a for a in range(D,Nx-D,Lout)]
    y = [a for a in range(D,Ny-D,Lout)]
    z = [a for a in range(D,Nz-D,Lout)]

    if x[-1]+Lin-D>=Nx-1:
        del x[-1]
    if y[-1]+Lin-D>=Ny-1:
        del y[-1]
    if z[-1]+Lin-D>=Nz-1:
        del z[-1]

    submaps_dict_4D = {}

    for in_x in x:
        for in_y in y:
            for in_z in z:
                submap_3D = np.copy(inp_map[in_x-D:in_x+Lin-D, in_y-D:in_y+Lin-D, in_z-D:in_z+Lin-D])
                sublable_3D = np.copy(inp_label[in_x:in_x+Lout ,in_y:in_y+Lout,in_z:in_z+Lout])
                sun_norm_3D = normalize_3D_box(submap_3D, mean=MEAN, sigma = SIGMA)
                sun_norm_exp_4D = np.expand_dims(sun_norm_3D,0)
                submaps_dict_4D[(in_x,in_y,in_z)] = {"em_map":sun_norm_exp_4D, "label":sublable_3D}


    return submaps_dict_4D

def maps_to_submaps_4D(inp_map, Lin=None, Lout=None):
    Nch, Nx, Ny, Nz = inp_map.shape
    D = (Lin-Lout)//2
    x = [a for a in range(D,Nx-D,Lout)]
    y = [a for a in range(D,Ny-D,Lout)]
    z = [a for a in range(D,Nz-D,Lout)]

    if x[-1]+Lin-D>=Nx-1:
        del x[-1]
    if y[-1]+Lin-D>=Ny-1:
        del y[-1]
    if z[-1]+Lin-D>=Nz-1:
        del z[-1]

    submaps_dict_4D = {}

    for in_x in x:
        for in_y in y:
            for in_z in z:
                submap_4D = np.copy(inp_map[:,in_x-D:in_x+Lin-D, in_y-D:in_y+Lin-D, in_z-D:in_z+Lin-D])
                submaps_dict_4D[(in_x,in_y,in_z)] = submap_4D
    return submaps_dict_4D


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
        sample ={"em_map": em_map, "label":labels_map, \
            "file_name": label_file, "un_norm_map":un_norm_map }
        return sample

