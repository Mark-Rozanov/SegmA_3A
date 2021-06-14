import numpy as np
import chimera
from chimera import runCommand
import sys
from Commands import CommandError

from VolumeViewer import open_volume_file
import time

label_dict = {"ALA":1,"ARG":2,"ASN":3,"ASP":4,"CYS":5,"CYH":5,"CYD":5,\
"GLN":6,"GLU":7,"GLY":8,"HIS":9,"ILE":10,\
"LEU":11,"LYS":12,"MET":13,"PHE":14,"PRO":15,"TPR":15,"CPR":15,\
"SER":16,"THR":17,"TRP":18,"TYR":19,"VAL":20,"NONE":0, "A":21,\
"C":21,"U":21,"G":21, "DA":21,"DC":21,"DU":21,"DG":21,"DU":21}

VX_SIZE=1 
RES = 3.0

def calc_voxalization_by_res_type(pdb_id,grid3D,res=3.0 , resTypes = []):

    Id_for_molmap = 5001

    res_map = {}

    ##loop on atoms
    for r_name in resTypes:
        # selection
        try: 
            runCommand('molmap #{}:{} {}  modelId {}'.\
                format(pdb_id,r_name,res,Id_for_molmap))
        except CommandError as e:
            print(r_name, e)
            runCommand('vop new zero_map origin {},{},{} modelId {}'.\
                    format(np.mean(grid3D[0]),np.mean(grid3D[1]),np.mean(grid3D[1]),Id_for_molmap))

                

        #extract matrix (copy?)
        res_map[r_name]=map_to_matrix(Id_for_molmap,grid3D)

        # delete mol map
        runCommand('close #{}'.format(Id_for_molmap))

    return res_map



def get_object_by_id(id):
    all_objs = filter(lambda x:x.id==id, chimera.openModels.list())
    if len(all_objs)==1:
        return all_objs[0]
    else:
        return -1


def map_to_matrix(map_id,grid3D):
    v_obj = get_object_by_id(map_id)
    Xs,Ys,Zs = grid3D
    xyz_coor = np.vstack((np.reshape(Xs,-1),np.reshape(Ys,-1),np.reshape(Zs,-1))).transpose()
    values, outside = v_obj.interpolated_values(xyz_coor,out_of_bounds_list = True)
    mtrx = np.reshape(values,Xs.shape)

    return mtrx

def calc_3D_grid_from_map(map_id,vx_size,margin):


    #extract grid
    v_obj = get_object_by_id(map_id)

    Xmin,Xmax = v_obj.xyz_bounds()
    Xmin = np.floor(Xmin)
    Xmax = np.ceil(Xmax)
    xr = np.arange(Xmin[0]-margin,Xmax[0]+margin,vx_size)
    yr = np.arange(Xmin[1]-margin,Xmax[1]+margin,vx_size)
    zr = np.arange(Xmin[2]-margin,Xmax[2]+margin,vx_size)
    Xs,Ys,Zs = np.meshgrid(xr,yr,zr,indexing='ij')


    #return output
    return (Xs,Ys,Zs)


def calc_3D_grid_from_pdb(pdb_id,vx_size,margin):

    syth_map_id = 6001
    #molmap
    res = vx_size*3.0
    runCommand('molmap #{} {} gridSpacing {} modelId {} replace false '\
               .format(pdb_id,res,vx_size,syth_map_id))

    Xs,Ys,Zs = calc_3D_grid_from_map(syth_map_id,vx_size,margin)

    #remove models
    runCommand('close #{}'.format(syth_map_id))
    #return output
    return (Xs,Ys,Zs)


def calc_all_matrices(pdb_file=None, map_file=None,vx_size = None, res = None, resTypesDict = None):

    map_obj = open_volume_file(map_file)[0]
    map_id = map_obj.id
    margin = vx_size*3


    if pdb_file != None:
        prot1 = chimera.openModels.open(pdb_file)[0]
        pdb_id = prot1.id
        #remove hydrogens
        runCommand('delete #{}@H'.format(pdb_id))
        grid3D = calc_3D_grid_from_pdb(pdb_id,vx_size,margin)
        res_mtrc = calc_voxalization_by_res_type(pdb_id,grid3D,res=res, resTypes = resTypesDict.keys())
        N_labels = np.max(resTypesDict.values())+1

        em_mtrx = map_to_matrix(map_id,grid3D)
        seg_matrix = np.zeros((N_labels, em_mtrx.shape[0], em_mtrx.shape[1], em_mtrx.shape[2]))
        seg_matrix[label_dict["NONE"],:,:,:] = 0.2
        for rs in res_mtrc.keys():
            seg_matrix[resTypesDict[rs],:,:,:] = np.maximum(seg_matrix[resTypesDict[rs],:,:,:], res_mtrc[rs])
        label_mtrx = np.argmax(seg_matrix, axis=0)


    else:
        grid3D = calc_3D_grid_from_map(map_id,vx_size,margin)
        em_mtrx = map_to_matrix(map_id,grid3D)
        label_mtrx = None

    return em_mtrx, label_mtrx


if __name__ == "chimeraOpenSandbox":

    if len(sys.argv)==5:
        ref_pdb_file=None
        map_file=sys.argv[3]
        out_image_file=sys.argv[4]
        em_mtrx, label_mtrx = calc_all_matrices(ref_pdb_file, map_file,vx_size = VX_SIZE, res = RES, resTypesDict = label_dict)
        np.save(out_image_file, em_mtrx)
    #    np.save(out_labels_file, label_mtrx)

    runCommand('stop')





