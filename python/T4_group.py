# Based on implementation from Bekkers (2020) - B-Spline CNNs on Lie Groups.

# Class implementation of the Lie group SE(2)
import torch
import numpy as np
import S4_group

T4_to_S4_list = [0,2,7,9,10,11,12,14,19,21,22,23]
T4_to_S4_dict = {}
S4_to_T4_dict = {}
for i,j in enumerate(T4_to_S4_list):
    T4_to_S4_dict[i] = j 
    S4_to_T4_dict[j] = i 

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

def rot_mat(xx,i):
    return S4_group.rot_mat(xx,T4_to_S4_dict[i]) 


def get_inverse_cayley():
    ict_S4 = S4_group.get_inverse_cayley()
    ict_T4 = np.zeros((len(T4_to_S4_list),len(T4_to_S4_list)),dtype = int)
    
    for i in range(len(T4_to_S4_list)):
        for j in range(len(T4_to_S4_list)):
            s4_i = T4_to_S4_dict[i]
            s4_j = T4_to_S4_dict[j]
            s4_val = ict_S4[s4_i,s4_j]
            t4_val = S4_to_T4_dict[s4_val]
            ict_T4[i,j] = np.int(t4_val)
    return ict_T4

## The sub-group H:
class H:
    # Label for the group
    name = 'SO(2)'
    # Dimension of the sub-group H
    n = 1  # Each element consists of 1 parameter
    # The identify element
    rots = [i for i in range(len(T4_to_S4_list))]
    inv_cayley_table = get_inverse_cayley()

    ## Essential for constructing the group G = R^n \rtimes H
    # Define how H acts transitively on R^n
    ## TODO: So far just for multiples of 90 degrees. No interpolation required
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
        
