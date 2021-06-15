
import torch
import layers3D
from layers3D import layer3D
import dbloader
import torch.nn as nn
import torch.nn.functional as F
import Klein4 
layers = layer3D(Klein4) # The layers is instantiated with the group structure as input

class BN_group(nn.Module):
    def __init__(self, Nout):
        torch.nn.Module.__init__(self)
        self.C = Nout*4
        self.bn = nn.BatchNorm3d(self.C)
    def forward(self, x):
        shape_in = x.shape
        shape_out=[shape_in[0], self.C, shape_in[-3],shape_in[-2],shape_in[-1] ]
        x = torch.reshape(x, shape_out)
        x = self.bn(x)
        x = torch.reshape(x, shape_in)
        return x
        

class Net(nn.Module):
    def __init__(self, net_name):
        super(Net, self).__init__()
        
        self.name = net_name
        
        self.conv1g = layers.ConvRnG(23,22,3, padding = 0).float()
        self.bn1  = BN_group(22)
        self.elu1 = nn.LeakyReLU()
        
        self.conv2g = layers.ConvGG(22,22,3, padding = 0).float()
        self.bn2  = BN_group(22)
        self.elu2 = nn.LeakyReLU()

        self.conv3g = layers.ConvGG(22,22,2, padding = 0).float()
        self.bn3  = BN_group(22)
        self.elu3 = nn.LeakyReLU()

        self.conv4g = layers.ConvGG(22,22,2, padding = 0).float()
        self.bn4  = BN_group(22)
        self.elu4 = nn.LeakyReLU()

        self.conv5g = layers.ConvGG(22,22,3, padding = 0).float()
        self.bn5  = BN_group(22)
        self.elu5 = nn.LeakyReLU()


        self.conv6g = layers.ConvGG(22,2,3, padding = 0).float()
        self.bn6  = BN_group(2)
        self.elu6 = nn.LeakyReLU()

        self.sftmx6 = nn.LogSoftmax(dim=1)


        
    def forward(self, x):

        x = self.conv1g(x)
        x = self.bn1(x)
        x = self.elu1(x)
        
        x = self.conv2g(x)
        x = self.bn2(x)
        x = self.elu2(x)

        x = self.conv3g(x)
        x = self.bn3(x)
        x = self.elu3(x)

        x = self.conv4g(x)
        x = self.bn4(x)
        x = self.elu4(x)

        x = self.conv5g(x)
        x = self.bn5(x)
        x = self.elu5(x)

        x = self.conv6g(x)
        x = self.bn6(x)
        x = self.elu6(x)

        x = torch.max(x,2)[0] 
        x = self.sftmx6(x)

        return x
    def test_map(self, input3Dmap, Lin = None, Lout = None):
        map_5D = input3Dmap.unsqueeze(0).unsqueeze(0)

        map_out_5D = self.forward(map_5D)

        _, labels_out_4D = torch.max(map_out_5D.data, 1)

        labels_out_3D = torch.squeeze(labels_out_4D)
        P =  (Lin-Lout)//2

        labels_out = torch.nn.functional.pad(labels_out_3D, (P,P,P,P,P,P), mode='constant', value=0)

        return labels_out
