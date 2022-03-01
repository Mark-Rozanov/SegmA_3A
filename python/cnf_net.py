import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import K4_group 
from layers3D import layer3D

layers = layer3D(K4_group) # The layers is instantiated with the group structure as input

N_map_chan = 1
N_seg_chan = 22
N_int_map = 5
N_interior_chan = 10
N_labels = 2

class BN_group(nn.Module):
    def __init__(self, Nout):
        torch.nn.Module.__init__(self)
        self.C = Nout*len(layers.H.rots)
        self.bn = nn.BatchNorm3d(self.C)
    def forward(self, x):
        shape_in = x.shape
        shape_out=[shape_in[0], self.C, shape_in[-3],shape_in[-2],shape_in[-1] ]
        x = torch.reshape(x, shape_out)
        x = self.bn(x)
        x = torch.reshape(x, shape_in)
        return x

class Net(nn.Module):
    def __init__(self, net_name, seg_nn = None):
        super(Net, self).__init__()

        self.Lin = 15
        self.seg_nn = seg_nn
        self.name = net_name
        self.Lout = self.Lin

        #convolution - real map
        self.conv1g_r = layers.ConvRnG(N_map_chan,N_int_map,4, padding = 0).float()
        self.bn1_r  = BN_group(N_int_map)
        self.elu1_r = nn.LeakyReLU()
        self.Lout = self.Lout-4+1
        
        self.conv2g_r = layers.ConvGG(N_int_map,N_int_map,4, padding = 0).float()
        self.bn2_r  = BN_group(N_int_map)
        self.elu2_r = nn.LeakyReLU()
        self.Lout = self.Lout-4+1

        self.conv3g_r = layers.ConvGG(N_int_map,N_int_map,5, padding = 0).float()
        self.bn3_r  = BN_group(N_int_map)
        self.elu3_r = nn.LeakyReLU()
        self.Lout = self.Lout-5+1

        #convolution all
        self.conv4g_r = layers.ConvRnG(N_int_map+N_seg_chan,N_interior_chan,3, padding = 0).float()
        self.bn4_r  = BN_group(N_interior_chan)
        self.elu4_r = nn.LeakyReLU()
        self.Lout = self.Lout-3+1

        self.conv5g_r = layers.ConvGG(N_interior_chan,N_interior_chan,2, padding = 0).float()
        self.bn5_r  = BN_group(N_interior_chan)
        self.elu5_r = nn.LeakyReLU()
        self.Lout = self.Lout-2+1

        self.conv6g_r = layers.ConvGG(N_interior_chan,N_interior_chan,2, padding = 0).float()
        self.bn6_r  = BN_group(N_interior_chan)
        self.elu6_r = nn.LeakyReLU()
        self.Lout = self.Lout-2+1


        #deconvolution - segmented
        self.de_conv1 = torch.nn.ConvTranspose3d(N_interior_chan,N_interior_chan,2, padding = 0).float()
        self.bn_d1  = nn.BatchNorm3d(N_interior_chan)
        self.elu_d1 = nn.LeakyReLU()
        self.Lout = self.Lout+2-1
        
        self.de_conv2 = torch.nn.ConvTranspose3d(N_interior_chan+N_interior_chan,N_interior_chan,2, padding = 0).float()
        self.bn_d2  = nn.BatchNorm3d(N_interior_chan)
        self.elu_d2 = nn.LeakyReLU()
        self.Lout = self.Lout+2-1

        self.de_conv3 = torch.nn.ConvTranspose3d(N_interior_chan+N_interior_chan,N_labels,3, padding = 0).float()
        self.bn_d3  = nn.BatchNorm3d(N_labels)
        self.elu_d3 = nn.LeakyReLU()
        self.Lout = self.Lout+3-1

        self.sftmx_cnf = nn.Softmax(dim=1)

    def forward(self, x):
        #convolution of real map
        #INITIAL LENGTH - 15
        x_r = self.conv1g_r(x)
        x_r = self.bn1_r(x_r)
        x_r = self.elu1_r(x_r)
        #LENGTH - 12
        x_r = self.conv2g_r(x_r)
        x_r = self.bn2_r(x_r)
        x_r = self.elu2_r(x_r)
        #LENGTH - 9
        x_r = self.conv3g_r(x_r) 
        x_r = self.bn3_r(x_r)
        x_r = self.elu3_r(x_r)
        #LENGTH - 5
        x_r = torch.max(x_r,2)[0] 


        #convultion of real with segmentation
        with torch.no_grad():
            x_seg_0, x_clf_0 = self.seg_nn(x)
        x_all = torch.cat((x_seg_0,x_r),1)

        #LENGTH - 5 
        x_all1 = self.conv4g_r(x_all)
        x_all1 = self.bn4_r(x_all1)
        x_all1 = self.elu4_r(x_all1)
        x_all1_m = torch.max(x_all1,2)[0] 

        #LENGTH - 3
        x_all2 = self.conv5g_r(x_all1)
        x_all2 = self.bn5_r(x_all2)
        x_all2 = self.elu5_r(x_all2)
        x_all2_m = torch.max(x_all2,2)[0] 

        #LENGTH - 2
        x_all3 = self.conv6g_r(x_all2)
        x_all3 = self.bn6_r(x_all3)
        x_all3 = self.elu6_r(x_all3)
        #LENGTH - 1
        x_all3_m = torch.max(x_all3,2)[0] 

        #deconvolution
        xd_1 = self.de_conv1(x_all3_m)
        xd_1 = self.bn_d1(xd_1)
        xd_1 = self.elu_d1(xd_1)
        #LENGTH - 2
        xd_2 = torch.cat((xd_1,x_all2_m),1)
        xd_2 = self.de_conv2(xd_2)
        xd_2 = self.bn_d2(xd_2)
        xd_2 = self.elu_d2(xd_2)
        #LENGtH - 3
        xd_3 = torch.cat((xd_2,x_all1_m),1)
        xd_3 = self.de_conv3(xd_3)
        xd_3 = self.bn_d3(xd_3)
        xd_3 = self.elu_d3(xd_3)
        #LENGTH - 5 
        
        cnf_labels = self.sftmx_cnf(xd_3)
        
        return cnf_labels, x_seg_0, x_clf_0

    def get_trained_parameters(self):
        all_params = [x for x in self.parameters()]
        seg_params = [x for x in self.seg_nn.parameters()]
        trained_parameters = list(set(all_params)-set(seg_params))
        return trained_parameters   