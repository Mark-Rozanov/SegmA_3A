import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
N_map_chan = 1
N_seg_chan = 22
N_int_map = 10
N_interior_chan = 40
N_atoms = 6

class Net(nn.Module):
    def __init__(self, net_name, clf_net = None):
        super(Net, self).__init__()

        self.Lin = 15
        self.Lout = 5

        self.name = net_name
        self.clf_net = clf_net

        #convolution - real map
        self.conv1_r = torch.nn.Conv3d(N_map_chan,N_int_map,4, padding = 0).float()
        self.bn1_r  = nn.BatchNorm3d(N_int_map)
        self.elu1_r = nn.LeakyReLU()

        self.conv2_r = torch.nn.Conv3d(N_int_map,N_int_map,4, padding = 0).float()
        self.bn2_r  = nn.BatchNorm3d(N_int_map)
        self.elu2_r = nn.LeakyReLU()

        self.conv3_r = torch.nn.Conv3d(N_int_map,N_int_map,5, padding = 0).float()
        self.bn3_r  = nn.BatchNorm3d(N_int_map)
        self.elu3_r = nn.LeakyReLU()

        #convolution all
        self.conv4 = torch.nn.Conv3d(N_int_map+N_seg_chan,N_interior_chan,3, padding = 0).float()
        self.bn4  = nn.BatchNorm3d(N_interior_chan)
        self.elu4 = nn.LeakyReLU()

        self.conv5 = torch.nn.Conv3d(N_interior_chan,N_interior_chan,2, padding = 0).float()
        self.bn5  = nn.BatchNorm3d(N_interior_chan)
        self.elu5 = nn.LeakyReLU()

        self.conv6 = torch.nn.Conv3d(N_interior_chan,N_interior_chan,2, padding = 0).float()
        self.bn6  = nn.BatchNorm3d(N_interior_chan)
        self.elu6 = nn.LeakyReLU()

        #deconvolution - segmented
        self.de_conv1_at = torch.nn.ConvTranspose3d(N_interior_chan,N_interior_chan,2, padding = 0).float()
        self.bn_d1_at  = nn.BatchNorm3d(N_interior_chan)
        self.elu_d1_at = nn.LeakyReLU()
        
        self.de_conv2_at = torch.nn.ConvTranspose3d(N_interior_chan+N_interior_chan,N_interior_chan,2, padding = 0).float()
        self.bn_d2_at  = nn.BatchNorm3d(N_interior_chan)
        self.elu_d2_at = nn.LeakyReLU()

        self.de_conv3_at = torch.nn.ConvTranspose3d(N_interior_chan+N_interior_chan,N_atoms,3, padding = 0).float()
        self.bn_d3_at  = nn.BatchNorm3d(N_atoms)
        self.elu_d3_at = nn.LeakyReLU()
 
        self.sftmx = nn.Softmax(dim=1)

    def forward(self, x):

        #convolution of real map
        #INITIAL LENGTH - 15
        x_r = self.conv1_r(x)
        x_r = self.bn1_r(x_r)
        x_r = self.elu1_r(x_r)
        #LENGTH - 12
        x_r = self.conv2_r(x_r)
        x_r = self.bn2_r(x_r)
        x_r = self.elu2_r(x_r)
        #LENGTH - 9
        x_r = self.conv3_r(x_r) 
        x_r = self.bn3_r(x_r)
        x_r = self.elu3_r(x_r)
        #LENGTH - 5

        #convultion of real with segmentation
        xs_0 = self.clf_net(x)
        x_all = torch.cat((xs_0,x_r),1)

        #LENGTH - 5 
        x_all1 = self.conv4(x_all)
        x_all1 = self.bn4(x_all1)
        x_all1 = self.elu4(x_all1)
        #LENGTH - 3
        x_all2 = self.conv5(x_all1)
        x_all2 = self.bn5(x_all2)
        x_all2 = self.elu5(x_all2)
        #LENGTH - 2
        x_all3 = self.conv6(x_all2)
        x_all3 = self.bn6(x_all3)
        x_all3 = self.elu6(x_all3)
        #LENGTH - 1

        #deconvolution
        xd_1 = self.de_conv1_at(x_all3)
        xd_1 = self.bn_d1_at(xd_1)
        xd_1 = self.elu_d1_at(xd_1)
        #LENGTH - 2
        xd_2 = torch.cat((xd_1,x_all2),1)
        xd_2 = self.de_conv2_at(xd_2)
        xd_2 = self.bn_d2_at(xd_2)
        xd_2 = self.elu_d2_at(xd_2)
        #LENGtH - 3
        xd_3 = torch.cat((xd_2,x_all1),1)
        xd_3 = self.de_conv3_at(xd_3)
        xd_3 = self.bn_d3_at(xd_3)
        xd_3 = self.elu_d3_at(xd_3)
        #LENGTH - 5 
        
        out_atoms = self.sftmx(xd_3)
        
        return out_atoms