import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
N_map_chan = 1
N_seg_chan = 22
N_int_map = 10
N_interior_chan = 40
N_labels = 22
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
        self.de_conv1 = torch.nn.ConvTranspose3d(N_interior_chan,N_interior_chan,2, padding = 0).float()
        self.bn_d1  = nn.BatchNorm3d(N_interior_chan)
        self.elu_d1 = nn.LeakyReLU()
        
        self.de_conv2 = torch.nn.ConvTranspose3d(N_interior_chan+N_interior_chan,N_interior_chan,2, padding = 0).float()
        self.bn_d2  = nn.BatchNorm3d(N_interior_chan)
        self.elu_d2 = nn.LeakyReLU()

        self.de_conv3 = torch.nn.ConvTranspose3d(N_interior_chan+N_interior_chan,N_labels,3, padding = 0).float()
        self.bn_d3  = nn.BatchNorm3d(N_labels)
        self.elu_d3 = nn.LeakyReLU()
 
        self.sftmx_seg = nn.Softmax(dim=1)

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
        out_clf = self.clf_net(x)
        x_all = torch.cat((out_clf,x_r),1)

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
        xd_1 = self.de_conv1(x_all3)
        xd_1 = self.bn_d1(xd_1)
        xd_1 = self.elu_d1(xd_1)
        #LENGTH - 2
        xd_2 = torch.cat((xd_1,x_all2),1)
        xd_2 = self.de_conv2(xd_2)
        xd_2 = self.bn_d2(xd_2)
        xd_2 = self.elu_d2(xd_2)
        #LENGtH - 3
        xd_3 = torch.cat((xd_2,x_all1),1)
        xd_3 = self.de_conv3(xd_3)
        xd_3 = self.bn_d3(xd_3)
        xd_3 = self.elu_d3(xd_3)
        #LENGTH - 5 
        
        out_labels = self.sftmx_seg(xd_3)
        
        return out_labels, out_clf

    def get_trained_parameters(self):
        all_params = [x for x in self.parameters()]
        seg_params = [x for x in self.clf_net.parameters()]
        trained_parameters = list(set(all_params)-set(seg_params))
        return trained_parameters   