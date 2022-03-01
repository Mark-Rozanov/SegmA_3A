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



## The sub-group H:
class H:
    # Label for the group
    name = 'SO(2)'
    # Dimension of the sub-group H
    n = 1  # Each element consists of 1 parameter
    # The identify element
    rots = [i for i in range(4)]

    ## Essential for constructing the group G = R^n \rtimes H
    # Define how H acts transitively on R^n
    ## TODO: So far just for multiples of 90 degrees. No interpolation required
    def left_representation_on_Rn(h, xx,bb):
        if h == 0:
            xx_new= xx
        elif h == 1:
            xx_new = torch.rot90(xx, k=2, dims=[-2, -3])
        elif h == 2:
            xx_new = torch.rot90(xx, k=2, dims=[-1, -3])
        elif h == 3:
            xx_new = torch.rot90(xx, k=2, dims=[-1, -2])
        
        return xx_new,bb

        
    def left_representation_on_G(h, fx,bb):
        cayley = np.asarray([[0,1,2,3],
                             [1,0,3,2],
                             [2,3,0,1],
                             [3,2,1,0]])
        p = cayley[h]
        kernel=[]
        for i in p:
            w,b = H.left_representation_on_Rn(h, fx[i],bb)
            kernel.append(w) 
        kernel = torch.cat(kernel, dim=2)
        
        return kernel,bb[0]
        
