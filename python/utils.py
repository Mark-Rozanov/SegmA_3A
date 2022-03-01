import numpy as np

AA_BoxIn = 11
AA_BoxOut = 1
AA_HALF_BOX = (AA_BoxIn-AA_BoxOut)//2
AA_LIN =AA_BoxIn*4
AA_LOUT=AA_LIN - (AA_BoxIn-AA_BoxOut)

label_dict = {"ALA":1,"ARG":2,"ASN":3,"ASP":4,"CYS":5,"CYH":5,"CYD":5,\
"GLN":6,"GLU":7,"GLY":8,"HIS":9,"ILE":10,\
"LEU":11,"LYS":12,"MET":13,"PHE":14,"PRO":15,"TPR":15,"CPR":15,\
"SER":16,"THR":17,"TRP":18,"TYR":19,"VAL":20,"NONE":0, "A":21,\
"C":21,"U":21,"G":21, "DA":21,"DC":21,"DU":21,"DG":21,"DU":21,"NCL":21}

element_dict = {"NONE":0,"CA":1,"CB":2, "C":3,"N":4,"O":5}

def acc_by_type_batch(output, y_true, l_dict):
    predicted = np.argmax(output, 1)

    
    res_dict={}
    for ky in l_dict.keys():
        res_dict[ky]={}
        true_1 = y_true==l_dict[ky]
        true_0 = y_true!=l_dict[ky]

        pred_1 = predicted==l_dict[ky]
        pred_0 = predicted!=l_dict[ky]

        res_dict[ky]["tp"] = np.float(np.sum(np.logical_and(pred_1, true_1)))
        res_dict[ky]["fp"] = np.float(np.sum(np.logical_and(pred_1, true_0)))
        res_dict[ky]["fn"] = np.float(np.sum(np.logical_and(pred_0, true_1)))

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

    submaps_dict = {}

    for in_x in x:
        for in_y in y:
            for in_z in z:
                submap_3D = np.copy(inp_map[in_x-D:in_x+Lin-D, in_y-D:in_y+Lin-D, in_z-D:in_z+Lin-D])
                sublable_3D = np.copy(inp_label[in_x:in_x+Lout ,in_y:in_y+Lout,in_z:in_z+Lout])
                submaps_dict[(in_x,in_y,in_z)] = {"em_map":submap_3D, "label":sublable_3D}

    return submaps_dict


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

class Atom():


    def get_pdb_string(self, atom_num, res_num=None, chain = "A"):
        if res_num==None:
            res_num = atom_num
        string_line = "ATOM  AAAAA  BBB CCC CHAIN DDDD     XXXXXXX YYYYYYY ZZZZZZZ  0.25 14.49           ELEMENT_TYPE"
        s = string_line
        s=s.replace('AAAAA', str(atom_num).ljust(4))
        s=s.replace('BBB', self.atom_type.ljust(3))
        s=s.replace('CCC', self.res_type.ljust(3))
        s=s.replace('CHAIN', chain)
        s=s.replace('DDDD', str(res_num).ljust(4))

        s=s.replace('XXXXXXX', '{:3.4g}'.format(self.x).ljust(7))
        s=s.replace('YYYYYYY', '{:3.4g}'.format(self.y).ljust(7))
        s=s.replace('ZZZZZZZ', '{:3.4g}'.format(self.z).ljust(7))

        s=s.replace('ELEMENT_TYPE', self.atom_type[0])

        return s
 
    def __init__(self, input_string):
        words = input_string.strip("\n").split()
        print("DEBUG words", words)
        error
        self.atom_type = words[0]
        self.x = float(words[1])
        self.y = float(words[2])
        self.z = float(words[3])
        self.res_type = words[4]
        self.res_num = int(words[5])

        self.true_prob = float(words[5])
        self.label_probs = np.zeros(22)
        for k in len(22): 
            self.labels_prob[k] = float(words[6+k])
        return

    def copy(self):


        return Atom(atom_type = self.atom_type,\
         res_type =self.res_type, res_num = self.res_num,x=self.x, y=self.y, z=self.z)


    def __init__(self, atom_type = "C", res_type ="ALA",res_num=None,x=0, y=0, z=0):

        self.atom_type = atom_type
        self.res_type = res_type
        self.x = x
        self.y = y
        self.z = z
        self.true_prob = 1.0
        self.label_probs = np.zeros(22)
        self.res_num = res_num
        return

    def get_string(self):
        atom_string = "TYPE XXXXXXX YYYYYYY ZZZZZZZ RES TRUE_PROB AA_NUM"
        s = atom_string
        s=s.replace('XXXXXXX', '{:3.4g}'.format(self.x).ljust(7))
        s=s.replace('YYYYYYY', '{:3.4g}'.format(self.y).ljust(7))
        s=s.replace('ZZZZZZZ', '{:3.4g}'.format(self.z).ljust(7))

        s=s.replace('TYPE', self.atom_type)
        s=s.replace('RES', self.res_type)
        s=s.replace('TRUE_PROB', '{:0.2g}'.format(self.true_prob).ljust(3))
        s=s.replace('AA_NUM', str(self.res_num).ljust(4))

        for l in self.label_probs:
            s = s + ' {:0.2g}'.format(l).ljust(3)
        return s

def write_pdb_file(file_name,atoms_list):
    with open(file_name,"w") as f_pdb:
        if len(atoms_list)==0:
            return
        elif len(atoms_list)==1:
            f_pdb.write(atoms_list[0].get_pdb_string(1,1))
        else:
            for (n_at,at) in enumerate(atoms_list[:-1]):
                f_pdb.write(at.get_pdb_string(n_at,n_at))
                f_pdb.write("\n")
            f_pdb.write(atoms_list[-1].get_pdb_string(n_at+1,n_at+1))

    return

def write_atoms_file(file_name,atoms_list ):
    with open(file_name,"w") as f_atoms:
        if len(atoms_list)==0:
            return
        elif len(atoms_list)==1:
            f_atoms.write(atoms_list[0].get_string())
        else:
            for at in atoms_list[:-1]:
                f_atoms.write(at.get_string())
                f_atoms.write("\n")
            f_atoms.write(atoms_list[-1].get_string())
    return

def atom_from_string(input_string):
    words = input_string.strip("\n").split()
    at  = Atom()
    at.atom_type = words[0]
    at.x = float(words[1])
    at.y = float(words[2])
    at.z = float(words[3])
    at.res_type = words[4]

    at.true_prob = float(words[5])
    at.label_probs = np.zeros(22)
    for k in range(22): 
        at.label_probs[k] = float(words[6+k])
    return at
