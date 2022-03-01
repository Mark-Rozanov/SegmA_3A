# Based on implementation from Bekkers (2020) - B-Spline CNNs on Lie Groups.

# Class implementation of the Lie group SE(2)
import torch
import numpy as np


# Rules for setting up a group class:
# A group element is always stored as a 1D vector, even if the elements consist
# only of a scalar (in which case the element is a list of length 1). Here we
# also assume that you can parameterize your group with a set of n parameters,
# with n the dimension of the group. The group elements are thus always lists of
# length n.
#
# This file requires the definition of the base/normal sub-group R^n and the
# sub-group H. Together they will define G = R^n \rtimes H.
#
# In order to derive G (it's product and inverse) we need for the group H to be
# known the group product, inverse and left action on R^n.
#
# Finally we need a way to sample the group. Therefore also a function "grid" is
# defined which samples the group as uniform as possible given a specified
# number of elements N. Not all groups allow a uniform sampling given an
# aribitrary N, in which case the sampling can be made approximately uniform by
# maximizing the distance between all sampled elements in H (e.g. via a
# repulsion model).


## The normal sub-group R^n:
# This is just the vector space R^n with the group product and inverse defined
# via the + and -.
class Rn:
    # Label for the group
    name = 'R^3'
    # Dimension of the base manifold N=R^n
    n = 3
    # The identity element
    e = torch.tensor([0., 0.,0], dtype=torch.float32)

def rot_basics(xx,i):
    if i==0:
        xx_new = xx
    elif i==1:
        xx_new = torch.rot90(xx, k=1, dims=[-1, -2])
    elif i==2:
        xx_new = torch.rot90(xx, k=2, dims=[-1, -2])
    elif i==3:
        xx_new = torch.rot90(xx, k=3, dims=[-1, -2])
    elif i==4:
        xx_new = torch.rot90(xx, k=1, dims=[-2, -3])
    elif i==5:
        xx_new = torch.rot90(xx, k=3, dims=[-2, -3])
    elif i==6:
        xx_new = torch.rot90(xx, k=1, dims=[-1, -3])
    elif i==7:
        x_6 = torch.rot90(xx, k=1, dims=[-1, -3])
        xx_new = torch.rot90(x_6, k=1, dims=[-1, -2])
    elif i==8:
        x_6 = torch.rot90(xx, k=1, dims=[-1, -3])
        xx_new = torch.rot90(x_6, k=2, dims=[-1, -2])
    elif i==9:
        x_6 = torch.rot90(xx, k=1, dims=[-1, -3])
        xx_new = torch.rot90(x_6, k=3, dims=[-1, -2])
    else:
        print(i)
        error
    return(xx_new)
    

def rot_mat(xx,i):
    if i==0:
        xx_new = rot_basics(xx,0)
    elif i==1:
        xx_new = rot_basics(xx,1)
    elif i==2:
        xx_new = rot_basics(xx,2)
    elif i==3:
        xx_new = rot_basics(xx,3)
    elif i==4:
        xx_new = rot_basics(xx,4)
    elif i==5:
        xx_new = rot_basics(xx,5)
    elif i==6:
        xx_new = rot_basics(xx,6)
    elif i==7:
        xx_new = rot_basics(xx,7)
    elif i==8:
        xx_new = rot_basics(xx,8)
    elif i==9:
        xx_new = rot_basics(xx,9)
    elif i==10:
        x_1 = rot_basics(xx,4)
        xx_new = rot_basics(x_1,3)
    elif i==11:
        x_1 = rot_basics(xx,1)
        xx_new = rot_basics(x_1,6)
    elif i==12:
        x_1 = rot_basics(xx,6)
        xx_new = rot_basics(x_1,6)
    elif i==13:
        x_1 = rot_basics(xx,6)
        xx_new = rot_basics(x_1,7)
    elif i==14:
        x_1 = rot_basics(xx,4)
        xx_new = rot_basics(x_1,4)
    elif i==15:
        x_1 = rot_basics(xx,6)
        xx_new = rot_basics(x_1,9)
    elif i==16:
        x_1 = rot_basics(xx,4)
        xx_new = rot_basics(x_1,2)
    elif i==17:
        x_1 = rot_basics(xx,5)
        xx_new = rot_basics(x_1,2)
    elif i==18:
        x_1 = rot_basics(xx,2)
        xx_new = rot_basics(x_1,8)
    elif i==19:
        x_1 = rot_basics(xx,1)
        xx_new = rot_basics(x_1,5)
    elif i==20:
        x_1 = rot_basics(xx,2)
        xx_new = rot_basics(x_1,6)
    elif i==21:
        x_1 = rot_basics(xx,3)
        xx_new = rot_basics(x_1,4)
    elif i==22:
        x_1 = rot_basics(xx,4)
        xx_new = rot_basics(x_1,1)
    elif i==23:
        x_1 = rot_basics(xx,1)
        xx_new = rot_basics(x_1,8)
    else:
        print(i)
        error

    return xx_new

def get_3Drotmat(x,y,z):
    c = [1.,0.,-1.,0.]
    s = [0.,1.,0.,-1]

    Rx = np.asarray([[c[x],     -s[x],  0.],
                     [s[x],     c[x],   0.],
                     [0.,       0.,     1.]])
    Ry = np.asarray([[c[y],     0.,     s[y]],
                     [0.,       1.,     0.],
                     [-s[y],    0.,     c[y]]])
    Rz = np.asarray([[1.,       0.,     0.],
                     [0.,       c[z],   -s[z]],
                     [0.,       s[z],   c[z]]])
    return Rz @ Ry @ Rx

def get_s4mat():
    Z = []
    for i in range(4):
        # Z_4 rotation about Y
        # S^2 rotation
        for j in range(4):
            z = get_3Drotmat(i,j,0)
            Z.append(z)
        # Residual pole rotations
        Z.append(get_3Drotmat(i,0,1))
        Z.append(get_3Drotmat(i,0,3))
    return Z


def get_cayleytable():
        Z = get_s4mat()
        cayley = []
        for y in Z:
            for z in Z:
                r = z @ y
                for i, el in enumerate(Z):
                    if np.sum(np.square(el - r)) < 1e-6:
                        cayley.append(i)
        cayley = np.stack(cayley)
        cayley = np.reshape(cayley, [24,24])
        return cayley

def get_inverse_cayley():
    
    ct = get_cayleytable()
    ct = np.transpose(ct)

    inv_cayley = np.zeros(ct.shape, dtype = int)
    I,J = ct.shape
    for i in range(I):
        for j in range(J):
                inv_cayley[i][ct[i][j]]=j
       
    return inv_cayley
## The sub-group H:
class H:
    # Label for the group
    name = 'SO(2)'
    # Dimension of the sub-group H
    n = 1  # Each element consists of 1 parameter
    # The identify element
    rots = [i for i in range(24)]
    
    inv_cayley_table= get_inverse_cayley()

    def left_representation_on_Rn(h, xx,bb):

        xx_new = rot_mat(xx,h)
        return xx_new,bb

        
    def left_representation_on_G(h, fx,bb):
        inv_cayley = H.inv_cayley_table
        kernel=[]
        for i in inv_cayley[h]:
            w,b = H.left_representation_on_Rn(h, fx[i],bb)
            kernel.append(w) 
        kernel = torch.cat(kernel, dim=2)
        
        
        return kernel,bb[0]
        
