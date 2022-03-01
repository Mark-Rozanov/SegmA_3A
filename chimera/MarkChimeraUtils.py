import chimera
import numpy as np
from VolumeData import Array_Grid_Data
from VolumeViewer import volume_from_grid_data
from VolumeViewer import active_volume
from VolumeViewer import open_volume_file
from numpy import zeros, array, dot, linalg
from chimera import runCommand
from Commands import CommandError
import utils
from utils import Atom, label_dict

def create_colouring_string(one_colors):
    clrs_scr = ""
    for oc in one_colors:
        #colors = colors + ["0,black:0.5,black:"
        clrs_scr = clrs_scr + oc
    #remove last colon
    clrs_scr=clrs_scr[:-1]
    return clrs_scr

def colors_by_aa_label():
    colors = []
    #NONE
    colors = colors + ["0,black:0.5,black:"]
    #"ALA":1,
    colors = colors + ["0.6,K:1.5,K:"]
    # "ARG":2
    colors = colors + ["1.6,Ca:2.5,Ca:"]
    # ,"ASN":3,
    colors = colors + ["2.6,Sc:3.5,Sc:"]
    # "ASP":4,
    colors = colors + ["3.6,Ti:4.5,Ti:"]
    # "CYS":5,"CYH":5,"CYD":5,
    colors = colors + ["4.6,V:5.5,V:"]
    #"GLN":6,
    colors = colors + ["5.6,Cr:6.5,Cr:"]
    # "GLU":7,
    colors = colors + ["6.6,Mn:7.5,Mn:"]
    # "GLY":8,
    colors = colors + ["7.6,Fe:8.5,Fe:"]
    # "HIS":9,
    colors = colors + ["8.6,Co:9.5,Co:"]
    # "ILE":10,
    colors = colors + ["9.6,Ni:10.5,Ni:"]
    #"LEU":11,
    colors = colors + ["10.6,O:11.5,O:"]
    # "LYS":12,
    colors = colors + ["11.6,Zn:12.5,Zn:"]
    # "MET":13,
    colors = colors + ["12.6,Ga:13.5,Ga:"]
    # "PHE":14,
    colors = colors + ["13.6,Ge:14.5,Ge:"]
    # "PRO":15, # "TPR":15,"CPR":15,
    colors = colors + ["14.6,As:15.5,As:"]
    #"SER":16
    colors = colors + ["15.6,Br:16.5,Br:"]
    # "THR":17,
    colors = colors + ["16.6,Kr:17.5,Kr:"]
    # "TRP":18,
    colors = colors + ["17.6,Lu:18.5,Lu:"]
    # "TYR":19,
    colors = colors + ["18.6,Cu:19.5,Cu:"]
    # "VAL":20,
    colors = colors + ["19.6,Au:20.5,Au:"]
    # #NULEOTIDES" "A":21,"C":21,"U":21,"G":21}
    colors = colors + ["20.5,N:21.5,N:"]

    return colors

def colors_by_true_false():
    colors = []
    #FALSE
    colors = colors + ["-0.5,red:0.5,red:"]
    #TRUE
    colors = colors + ["0.6,blue:1.5,blue:"]
    return colors


def create_map(mtrx, vx_size, org = [0.0,0.0,0.0], name = 'Marik'):

    step = [vx_size,vx_size,vx_size]
    mtrx_1 = np.swapaxes(mtrx,2,0)

    mtrx_1[np.where(mtrx_1==0)]=-0.01

    grid = Array_Grid_Data(mtrx_1, org, step, name = name)
    v = volume_from_grid_data(grid)

    return v

def map_to_matrix(map_id,grid3D):
    v_obj = get_object_by_id(map_id)
    Xs,Ys,Zs = grid3D
    xyz_coor = np.vstack((np.reshape(Xs,-1),np.reshape(Ys,-1),np.reshape(Zs,-1))).transpose()
    values, outside = v_obj.interpolated_values(xyz_coor,out_of_bounds_list = True)
    mtrx = np.reshape(values,Xs.shape)

    return mtrx

def get_residues_from_all_models(pdb_file):
    all_mdls = chimera.openModels.open(pdb_file,'PDB')
    res_list = []
    for mdl in all_mdls:
        res_list = res_list + mdl.residues
    return res_list

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


def calc_voxalization_by_atom_type(pdb_id,grid3D,res=3.0 , atomTypes = []):

    Id_for_copy = 4001
    Id_for_molmap = 5001

    vx_map = {}

    ##loop on atoms
    for at_name in atomTypes:

        ##copy structure
        runCommand('combine #{} name  {}_atoms modelId {}'.\
                   format(pdb_id, at_name,Id_for_copy))
        ## delete atoms
        runCommand('delete #{}:@/element!={}'.format(Id_for_copy,at_name))
        ## run molmap
        no_atoms = get_object_by_id(Id_for_copy)==-1


        if no_atoms:
            runCommand('vop new zero_map origin {},{},{} modelId {}'.\
                       format(np.mean(grid3D[0]),np.mean(grid3D[1]),np.mean(grid3D[1]),Id_for_molmap))
        else:
            runCommand('molmap #{} {}  modelId {}'.\
                   format(Id_for_copy,res,Id_for_molmap))

        #extract matrix (copy?)
        vx_map[at_name]=map_to_matrix(Id_for_molmap,grid3D)

        # delete copied structure
        runCommand('close #{}'.format(Id_for_copy))
        # delete mol map
        runCommand('close #{}'.format(Id_for_molmap))

    return vx_map

def calc_selected_atoms(all_atoms, grid3D):

    def within_borders(at):
        if at.element.name not in list(utils.element_dict.keys()):
            return False
        if at.coord()[0] <Xmin or at.coord()[0] >Xmax:
            return False
        if at.coord()[1] <Ymin or at.coord()[1] >Ymax:
            return False
        if at.coord()[2] <Zmin or at.coord()[2] >Zmax:
            return False
        return True
    
    (Xs,Ys,Zs) = grid3D

    Xmin = np.min(Xs)
    Xmax = np.max(Xs)
    Ymin = np.min(Ys)
    Ymax = np.max(Ys)
    Zmin = np.min(Zs)
    Zmax = np.max(Zs)

    selected_atoms = [x for x in all_atoms if within_borders(x)]

    atoms_list = []
    for at in selected_atoms:
        if at.name in ["CA","CB"]:
            atom_type = at.name
        else:
            atom_type = at.element.name

        x = at.coord()[0]-Xmin
        y = at.coord()[1]-Ymin
        z = at.coord()[2]-Zmin
        atoms_list.append(Atom(atom_type = atom_type, res_type =at.residue.type, x=x, y=y, z=z))

    return atoms_list

def calc_all_matrices(pdb_file, map_file,margin_vx = None,vx_size = None, res = None, resTypesDict = None):
    prot1 = chimera.openModels.open(pdb_file)[0]
    map_obj = open_volume_file(map_file)[0]
    pdb_id = prot1.id
    map_id = map_obj.id

    #remove hydrogens
    runCommand('delete #{}@H'.format(pdb_id))
    margin = vx_size*margin_vx
    Xs,Ys,Zs = calc_3D_grid(pdb_id,vx_size,margin)
    grid3D = (Xs,Ys,Zs)
    em_mtrx = map_to_matrix(map_id,grid3D)

    res_mtrc = calc_voxalization_by_res_type(pdb_id,grid3D,res=res, resTypes = resTypesDict.keys())

    N_labels = np.max(resTypesDict.values())+1
    seg_matrix = np.zeros((N_labels, em_mtrx.shape[0], em_mtrx.shape[1], em_mtrx.shape[2]))
    seg_matrix[label_dict["NONE"],:,:,:] = 0.2
    for rs in res_mtrc.keys():
        seg_matrix[resTypesDict[rs],:,:,:] = np.maximum(seg_matrix[resTypesDict[rs],:,:,:], res_mtrc[rs])

    label_mtrx = np.argmax(seg_matrix, axis=0)
    dist_matrix = seg_matrix
    return em_mtrx, label_mtrx, dist_matrix


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
    return Xs,Ys,Zs


def calc_3D_grid(pdb_id,vx_size,margin):

    syth_map_id = 6001
    #molmap
    res = vx_size*3.0
    runCommand('molmap #{} {} gridSpacing {} modelId {} replace false '\
               .format(pdb_id,res,vx_size,syth_map_id))

    Xs,Ys,Zs = calc_3D_grid_from_map(syth_map_id,vx_size,margin)

    #remove models
    runCommand('close #{}'.format(syth_map_id))
    #return output
    return Xs,Ys,Zs


def get_rotamer_angles(res):
    angle_names = ["phi", "psi", "chi1", "chi2", "chi3", "chi4"]
    angles = np.array([res.phi, res.psi, res.chi1, res.chi2, res.chi3, res.chi4]).astype(float)
    angles[np.isnan(angles)] = -999
    ang_dict = {}
    for k in range(len(angle_names)):
        ang_dict[angle_names[k]] = angles[k]

    return ang_dict

def get_object_by_id(id):
    all_objs = filter(lambda x:x.id==id, chimera.openModels.list())
    if len(all_objs)==1:
        return all_objs[0]
    else:
        return -1

def getRotMatr2Res(res1,res2):
    x1,y1,z1 = getCB_CA_N_frame(res1)
    Rot1 = np.matrix([[x1[0], y1[0],z1[0]],
                         [x1[1], y1[1],z1[1]],
                         [x1[2], y1[2],z1[2]]])

    x2,y2,z2 = getCB_CA_N_frame(res2)
    Rot2 = np.matrix([[x2[0], y2[0],z2[0]],
                         [x2[1], y2[1],z2[1]],
                         [x2[2], y2[2],z2[2]]])



    Rot = np.transpose(Rot2)*Rot1
    return Rot, np.transpose(Rot1),np.transpose(Rot2)

def getCB_CA_N_frame(res1):

    #first res
    CA_atom = res1.findAtom('CA')
    CB_atom = res1.findAtom('CB')
    N_atom = res1.findAtom('N')

    x_axes = CB_atom.coord() - CA_atom.coord()
    y1_axes = N_atom.coord() - CA_atom.coord()


    z_axes = np.cross(x_axes,y1_axes)
    y_axes = -np.cross(x_axes,z_axes)
    x_axes = array(x_axes)
    x_axes = x_axes/np.linalg.norm(x_axes)
    y_axes = y_axes/np.linalg.norm(y_axes)
    z_axes = z_axes/np.linalg.norm(z_axes)

    #Rot = [x_axes'; y_axes';z_axes']
    return x_axes,y_axes,z_axes


def getResByNumInChain(prot, chainName, Num):
    for res in prot.residues:
        if res.id.chainId == chainName and res.id.position == Num:
            return res
    return []

def atomsList2spec(atomsList):
    spec = ""
    for atom in atomsList:
        mod = str(atom.molecule.id);
        sub_mod = str(atom.molecule.subid);
        chainId = atom.residue.id.chainId;
        resNum = str(atom.residue.id.position);
        atomType = atom.name
        altLoc = atom.altLoc

        spec = spec + ' ' +'#' +mod +'.' + sub_mod + ':' + resNum + '.' + chainId + '@' + atomType + '.' + altLoc

    return spec



def molmapCube(model_id, resolution):
    """
    creates map from model as the original mol map, but with cube grid
    """

    ####create grid and save as model
    #get gabarities
    min_x, max_x, min_y, max_y, min_z, max_z = getGabarities(model_id)
    #create empty cube map
    min_cor = min(min_x,min_y,min_z)
    max_cor = max(max_x,max_y,max_z)
    d_grid = resolution/3



    #run molmap
    molmap_com = 'molmap #'+str(model_id) + ' ' + str(resolution)+' gridSpacing ' + str(resolution/3.0)
    chimera.runCommand(molmap_com)
    map_orig = active_volume();

    # interpolation
    createCubeMapfromGivenMap(map_orig,min_cor, max_cor, d_grid)

    #delete the grid
    map_orig.destroy()



def getGabarities(model_id):
    #get list of atoms
    prot = chimera.openModels.list()[model_id]

    atom = prot.atoms[0]
    min_x = atom.coord()[0]
    max_x = atom.coord()[0]
    min_y = atom.coord()[1]
    max_y = atom.coord()[1]
    min_z = atom.coord()[2]
    max_z = atom.coord()[2]

    for atom in prot.atoms:
        x = atom.coord()[0]
        y = atom.coord()[1]
        z = atom.coord()[2]

        min_x = min(min_x,x)
        min_y = min(min_y,y)
        min_z = min(min_z,z)


        max_x = max(max_x,x)
        max_y = max(max_y,y)
        max_z = max(max_z,z)

    return min_x, max_x, min_y, max_y, min_z, max_z

def createCubeMapfromGivenMap(ref_map,min_cor, max_cor, d_grid):
    g= []
    for z_coor in drange(min_cor,max_cor,d_grid):
        y_list = [];
        for y_coor in drange(min_cor,max_cor,d_grid):
            x_list = []
            for x_coor in drange(min_cor,max_cor,d_grid):
                point_coordinate = (x_coor,y_coor,z_coor)
                x_list.append(point_coordinate);
            y_list.append(x_list);
        g.append(y_list)

    ga_3d = np.array(g);

    ga_shape = ga_3d.shape;
    ga_1d = np.reshape(ga_3d,[ga_shape[0]*ga_shape[1]*ga_shape[2], ga_shape[3]])

    # create original model
    map_region_model_1d =  ref_map.interpolated_values(ga_1d)
    map_region_model= np.reshape(map_region_model_1d,ga_shape[0:3])
    grid = Array_Grid_Data(map_region_model, (min_cor,min_cor,min_cor), (d_grid,d_grid,d_grid), name = 'EMRegion')
    v_orig = volume_from_grid_data(grid)

    return v_orig.id



    #calc SVD
    #plot

def drange(start, stop, step):
    r = start
    while r < stop:
     	yield r
     	r += step
