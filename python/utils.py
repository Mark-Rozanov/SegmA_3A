import numpy as np
import torch

label_dict = {"ALA":1,"ARG":2,"ASN":3,"ASP":4,"CYS":5,"CYH":5,"CYD":5,\
"GLN":6,"GLU":7,"GLY":8,"HIS":9,"ILE":10,\
"LEU":11,"LYS":12,"MET":13,"PHE":14,"PRO":15,"TPR":15,"CPR":15,\
"SER":16,"THR":17,"TRP":18,"TYR":19,"VAL":20,"NONE":0, "A":21,\
"C":21,"U":21,"G":21, "DA":21,"DC":21,"DU":21,"DG":21,"DU":21,"NCL":21}

def acc_by_type_batch(output, y_true):
    predicted = np.argmax(output, 1)

    
    res_dict={}
    for ky in label_dict.keys():
        res_dict[ky]={}
        true_1 = y_true==label_dict[ky]
        true_0 = y_true!=label_dict[ky]

        pred_1 = predicted==label_dict[ky]
        pred_0 = predicted!=label_dict[ky]


        res_dict[ky]["tp"] = np.sum(np.logical_and(pred_1, true_1))
        res_dict[ky]["fp"] = np.sum(np.logical_and(pred_1, true_0))
        res_dict[ky]["fn"] = np.sum(np.logical_and(pred_0, true_1))

    return res_dict 

def acc_batch(output, y_true):
        predicted = np.argmax(output, 1)
        true_predictions = predicted == y_true

        total = np.prod(y_true.shape)
        correct = np.sum(true_predictions)
        return correct/total

def normalize_3D_box(bx, mean=0, sigma = 1):
    bx_var = np.std(bx)
    #assert bx_var>0.001
    if bx_var<0.00000001:
        bx_norm = -999*np.ones(bx.shape)
    else:
        bx_norm = (bx-np.mean(bx))/bx_var*sigma+mean
    return bx_norm 


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
