import numpy as np
import chimera
from chimera import runCommand
import sys, os, glob, shutil
import traceback
sys.path.append("/home/temp/SegmA_3A/python/")
import MarkChimeraUtils
from VolumeViewer import open_volume_file
import time
#import utils_project
from VolumeData import Array_Grid_Data
from VolumeData import Grid_Data
import MoleculeMap
from MoleculeMap import molecule_map
import VolumeViewer
import VolumeData
from VolumeData import Array_Grid_Data
from VolumeData import Grid_Data
import MoleculeMap
from MoleculeMap import molecule_map
from Matrix import euler_xform
from utils import AA_BoxIn, AA_BoxOut, AA_HALF_BOX, AA_LIN, AA_LOUT
N_ANGLES = 10

from utils import Atom, maps_to_submaps, element_dict, write_atoms_file, write_pdb_file
from MarkChimeraUtils import calc_all_matrices

label_dict = {"ALA":1,"ARG":2,"ASN":3,"ASP":4,"CYS":5,"CYH":5,"CYD":5,\
"GLN":6,"GLU":7,"GLY":8,"HIS":9,"ILE":10,\
"LEU":11,"LYS":12,"MET":13,"PHE":14,"PRO":15,"TPR":15,"CPR":15,\
"SER":16,"THR":17,"TRP":18,"TYR":19,"VAL":20,"NONE":0, "A":21,\
"C":21,"U":21,"G":21, "DA":21,"DC":21,"DU":21,"DG":21,"DU":21}

ELEMENT_TYPES = list(element_dict.keys())

vx_size=1
margin = vx_size*3
resolution = 3.0
ATOMS_PATCH_THR=00
N_PAD = 24

def process_map_folder(rot_dir):

    def within_borders(at, ky,merge=3):

        in_x, in_y, in_z = ky

        Xmin = in_x*vx_size-merge
        Xmax = (in_x+AA_LOUT)*vx_size+merge
        Ymin = in_y*vx_size-merge
        Ymax = (in_y+AA_LOUT)*vx_size+merge
        Zmin = in_z*vx_size-merge
        Zmax = (in_z+AA_LOUT)*vx_size+merge


        if at.x <Xmin or at.x >Xmax:
            return False
        if at.y <Ymin or at.y >Ymax:
            return False
        if at.z <Zmin or at.z >Zmax:
            return False
        return True


    ## CREATE LABELS ##
    out_image_file = rot_dir+'/'+"input_map.npy"
    out_labels_file =  rot_dir+'/'+"true_label.npy"
    dist_file =  rot_dir+'/'+"dist_to_atoms.npy"

    inp_pdb_file = rot_dir+'/'+"prot.pdb"
    inp_map_file = rot_dir+'/'+"raw_map.mrc"

    em_mtrx, label_mtrx, dist_mtrx = calc_all_matrices(inp_pdb_file, inp_map_file,margin_vx = 25,vx_size = vx_size, res = resolution, resTypesDict = label_dict)
    np.save(out_image_file, em_mtrx)
    np.save(out_labels_file, label_mtrx)
    np.save(dist_file, dist_mtrx)

    ## CREATE PATCHES
    patch_dir = rot_dir+"/test_patches/"
    if  os.path.exists(patch_dir):
        shutil.rmtree(patch_dir)
    os.mkdir(patch_dir)
    image_file = rot_dir+'/'+"input_map.npy"
    labels_file =  rot_dir+'/'+"true_label.npy"
    map_file_no_norm = np.load(image_file)
    labels_map = np.load(labels_file)
    submap_dict = maps_to_submaps(map_file_no_norm, labels_map, Lin=AA_LIN, Lout=AA_LOUT)
    keys_all = [k for k in submap_dict.keys()]
    for ky in keys_all:
        D = (AA_LIN-AA_LOUT)//2

        patch_map_file = patch_dir + "X{}_Y{}_Z{}_map.npy".format(ky[0],ky[1],ky[2])
        patch_label_file = patch_dir + "X{}_Y{}_Z{}_label.npy".format(ky[0],ky[1],ky[2])

        np.save(patch_map_file, submap_dict[ky]["em_map"])
        np.save(patch_label_file, submap_dict[ky]["label"])

    return

if __name__ == "chimeraOpenSandbox" :

    map_folder=sys.argv[3]

    patches_dir = map_folder + '/test_patches/'

    if  os.path.exists(patches_dir):
        shutil.rmtree(patches_dir)
    os.mkdir(patches_dir)


    runCommand('close all')
    process_map_folder(map_folder)

    runCommand('stop')


def test_patch():
    runCommand('close all')
    rot_dir=('//home/iscb/wolfson/Mark/data/DomainShift/db/raw_from_web/350/EMD-12586/rot0')
    process_rotation_folder(rot_dir)

    runCommand('close all')
    volume_from_npy("/Users/markroza/Documents/GitHub/work_from_home/DomainShift/rot/patches_10/X39_Y39_Z39_map.npy")
    prt = chimera.openModels.open("/Users/markroza/Documents/GitHub/work_from_home/DomainShift/rot/patches_10/X39_Y39_Z39.pdb")[0]
    D = (AA_LIN-AA_LOUT)//2

    for at in prt.atoms:
        p = at.coord()
        p[0] = p[0] + D
        p[1] = p[1] + D
        p[2] = p[2] + D
        at.setCoord(p)
