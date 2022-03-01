
import os,sys
import torch
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from utils import element_dict, Atom


try:
    from torch import Tensor
except:
    "print NO TORCH"



import scipy
from scipy import ndimage
from scipy.signal import argrelextrema

WIDTH=0.5

def test_functionality():
    step=1
    X_min = -11
    X_max = 11
    Y_min = -10
    Y_max = 11
    Z_min = -5
    Z_max = 15
    list_centers = [(-2,7,-1),(4,5,0),(7,8,6)]
    x_vec, y_vec,z_vec, signal_inp = get_3D_mtrx(X_min,X_max,Y_min,Y_max,Z_min, Z_max,step,list_centers)
    pos, vals = get_peaks(signal_inp)
    print(pos,vals)
    return


def get_3D_mtrx(Xmin,Xmax,Ymin,Ymax,Zmin,Zmax,step,list_centers):
    x_vec = np.arange(Xmin, Xmax, step)    
    y_vec = np.arange(Ymin, Ymax, step)
    z_vec = np.arange(Zmin, Zmax, step)

    x_mesh, y_mesh, z_mesh = np.meshgrid(x_vec, y_vec, z_vec,indexing='ij')
    
    mtrx = x_mesh*0
    
    for pnt in list_centers:
        X0,Y0,Z0 = pnt
        dst2 = (x_mesh-X0)*(x_mesh-X0)+(y_mesh-Y0)*(y_mesh-Y0)+(z_mesh-Z0)*(z_mesh-Z0)
        val = np.exp(-dst2/WIDTH)
        mtrx = mtrx+val
    mtrx=mtrx    
    return x_vec, y_vec, z_vec, mtrx


def get_peaks(signal_inp, peaks_thr = -99):
    c_tmp = np.ones((3,3,3))
    signal_tns = torch.Tensor(signal_inp)
    kernel_tns = torch.Tensor(c_tmp)
    
    pd = kernel_tns.size(0)//2
    sig_pad = torch.nn.functional.pad(signal_tns.unsqueeze(0).unsqueeze(0),[pd,pd,pd,pd,pd,pd],mode='constant', value=0)
    out_tns = torch.nn.functional.conv3d(sig_pad, kernel_tns.unsqueeze(0).unsqueeze(0),padding= (0,0,0))
    
    out_np = out_tns.numpy().squeeze()/np.sum(c_tmp)
    
    x_peaks = argrelextrema(out_np,np.greater,axis =0,order=1)
    y_peaks = argrelextrema(out_np,np.greater,axis =1,order=1)
    z_peaks = argrelextrema(out_np,np.greater,axis =2,order=1)
    
    y_peaks = set(zip(y_peaks[0],y_peaks[1],y_peaks[2]))
    x_peaks = set(zip(x_peaks[0],x_peaks[1],x_peaks[2]))
    z_peaks = set(zip(z_peaks[0],z_peaks[1],z_peaks[2]))
    xyz_peaks = x_peaks.intersection(y_peaks).intersection(z_peaks)
    
    pos = []
    vals = []
    for pk in xyz_peaks:
        if out_np[pk[0],pk[1],pk[2]]>peaks_thr:
            pos.append((pk[0],pk[1],pk[2]))
            vals.append(out_np[pk[0],pk[1],pk[2]])
    
    return pos, vals
    
def extract_atoms(labels_map, TF_map, atoms_map, L_out, peaks_thr = -99):
    atoms_list = []

    for atom_type in element_dict.keys():
        atom_num = element_dict[atom_type]
        input_signal = atoms_map[atom_num,:,:,:]
        peaks, vals = get_peaks(input_signal)
        for atom_xyz in peaks:
            x,y,z = atom_xyz

            tf_prob = TF_map[1,x,y,z]
            labels_prob = labels_map[:,x,y,z].copy()

            at = Atom(atom_type = atom_type, x=x, y=y, z=z)
            at.true_prob = tf_prob
            at.label_probs = labels_prob

            atoms_list.append(at)

    return atoms_list



