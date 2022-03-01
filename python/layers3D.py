# Based on implementation from Bekkers (2020) - B-Spline CNNs on Lie Groups.

import torch
import numpy as np
import math

# Start of (Parent Class)
class layer3D(torch.nn.Module):
    def __init__(self, group):
        super(layer3D, self).__init__()
        self.group = group
        self.Rn = group.Rn
        self.H = group.H

    # TODO include dilation everywhere
    # Creates an spatial_layer object
    def ConvRnRn(self,
                 # Required arguments
                 N_in,              # Number of input channels
                 N_out,             # Number of output channels
                 kernel_size,       # Kernel size (integer)
                 # Optional generic arguments
                 stride=1,          # Spatial stride in the convolution
                 padding=1,         # Padding type
                 dilation=1,        # Dilation
                 conv_groups = 1,
                 wscale=1.0):       # White scaling
        return ConvRnRnLayer(self.group, N_in, N_out, kernel_size, stride, padding, dilation, conv_groups, wscale)

    # Creates a lifting_layer object
    def ConvRnG(
            self,
            # Required arguments
            N_in,                   # Number of input channels
            N_out,                  # Number of output channels
            kernel_size,            # Kernel size (integer)
            stride=1,               # Spatial stride in the convolution
            padding=1,              # Padding type
            dilation=1,             # Dilation
            conv_groups = 1,        # Name of generated tensorflow variables
            wscale = 1.0):          # White scaling
        return ConvRnGLayer(self.group, N_in, N_out, kernel_size, stride, padding, dilation, conv_groups, wscale)

    # Creates a group convolution layer object
    def ConvGG(
            self,
            # Required arguments
            N_in,                   # Number of input channels
            N_out,                  # Number of output channels
            kernel_size,            # Kernel size (integer)
            stride=1,               # Spatial stride in the convolution
            padding=1,              # Padding type
            dilation=1,             # Dilation
            conv_groups=1,          # Name of generated tensorflow variables
            wscale = 1.0):          # White scaling
        return ConvGGLayer(self.group, N_in, N_out, kernel_size, stride, padding, dilation, conv_groups, wscale)
##########################################################################
############################## ConvRnRnLayer #############################
##########################################################################
class ConvRnRnLayer(torch.nn.Module):
    def __init__(self,
                 group,
                 N_in,
                 N_out,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 conv_groups,
                 wscale):
        super(ConvRnRnLayer, self).__init__()
        ## Assert and set inputs
        self.kernel_type = 'Rn'
        self._assert_and_set_inputs(group, N_in, N_out, kernel_size, stride, padding, dilation, conv_groups)
        ## Construct the trainable weights and initialize them
        self.weights = torch.nn.ParameterList([torch.nn.Parameter(torch.Tensor(self.N_out, self.N_in, kernel_size, kernel_size,kernel_size))])
        self.biases = torch.nn.ParameterList([torch.nn.Parameter(torch.Tensor(self.N_out, self.N_in))])
        self._reset_parameters(wscale=wscale)

########################### Assert and set inputs ########################

    def _assert_and_set_inputs(self, group, N_in, N_out, kernel_size, stride, padding, dilation, conv_groups):
        self._assert_and_set_inputs_RnRn(group, N_in, N_out, kernel_size, stride, padding, dilation, conv_groups)

    def _assert_and_set_inputs_RnRn(self, group, N_in, N_out, kernel_size, stride, padding, dilation, conv_groups):
        ## Check (and parse) all the inputs
        # Include the dictionary of the used parent class
        self.group = group
        self.H = group.H
        self.Rn = group.Rn
        # Mandatory inputs
        self.N_in = self._assert_N_in(N_in)
        self.N_out = self._assert_N_out(N_out)
        self.kernel_size = self._assert_kernel_size(kernel_size)
        # Optional arguments
        self.conv_groups = self._assert_conv_groups(conv_groups)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def _assert_N_in(self, N_in):
        assert isinstance(N_in, int), "The specified argument \"N_in\" should be an integer."
        return N_in

    def _assert_N_out(self, N_out):
        assert isinstance(N_out, int), "The specified argument \"N_out\" should be an integer."
        return N_out

    def _assert_kernel_size(self, kernel_size):
        assert isinstance(kernel_size, int), "The specified argument \"kernel_size\" should be an integer."
        return kernel_size

    def _assert_conv_groups(self, conv_groups):
        assert isinstance(conv_groups, int), "The specified argument \"conv_groups\" should be an integer."
        return conv_groups

############################ Compute the output ##########################

    ## Public functions
    def kernel_and_bias(self, h=0):
        # The transformation to apply
        # Sample the kernel on the (transformed) grid
        return  self.H.left_representation_on_Rn(h, self.weights[0], self.biases[0])

    def forward(self, input):
        return  self.conv_Rn_Rn(input)

    def conv_Rn_Rn(self, input):
        kr,bs = self.kernel_and_bias(h=0)
        output = torch.conv3d(input=input,
                              weight=kr,
                              bias= bs,
                              stride=self.stride,
                              padding=self.padding,
                              dilation=self.dilation,
                              groups=self.conv_groups)
        return output

    def _reset_parameters(self, wscale):
        n = self.N_in
        k = self.kernel_size ** 2
        n *= k
        stdv = wscale * (1. / math.sqrt(n))
        self.stdv = stdv
        for w in self.weights:
            w.data.uniform_(-stdv, stdv)
        for b in self.biases:
            b.data.uniform_(-stdv, stdv)
        


##########################################################################
############################## ConvRnGLayer ##############################
##########################################################################
# Start of lifting_layer class
class ConvRnGLayer(ConvRnRnLayer, torch.nn.Module):
    def __init__(self,
                 group,
                 N_in,
                 N_out,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 conv_groups,
                 wscale):
        #torch.nn.Module.__init__(self)
        torch.nn.Module.__init__(self)
        ## Assert and set inputs
        self.kernel_type = 'Rn'
        self._assert_and_set_inputs(group, N_in, N_out, kernel_size, stride, padding, dilation, conv_groups)
        ## Construct the trainable weights and initialize them
        self.weights = torch.nn.ParameterList([torch.nn.Parameter(torch.Tensor(self.N_out, self.N_in, kernel_size, kernel_size,kernel_size))])
        self.biases = torch.nn.ParameterList([torch.nn.Parameter(torch.Tensor(self.N_out, 1,1))])

        self._reset_parameters(wscale=wscale)

########################### Assert and set inputs ########################

    # Method overriding:
    def _assert_and_set_inputs(self, group, N_in, N_out, kernel_size, stride, padding, dilation, conv_groups):
        # Default Rn assertions
        self._assert_and_set_inputs_RnRn(group, N_in, N_out, kernel_size, stride, padding, dilation, conv_groups)
        # Specific initialization/assertion
        self.N_h = group.H.rots


############################ Compute the output ##########################

    # Method overriding:
    def forward(self, input):
        return self.conv_Rn_G(input)

    def conv_Rn_G(self, input):
        # Generate the full stack of convolution kernels (all transformed copies)
        kernel_stack=[]
        bias_stack=[]
        
        for i in self.N_h:
            w,b = self.kernel_and_bias(h=i)
            kernel_stack.append(w)
            bias_stack.append(b)
         
        kernel_stack = torch.cat(kernel_stack, dim=0)
        bias_stack = torch.squeeze(torch.cat(bias_stack, dim=0))

        output = torch.conv3d(
            input=input,
            weight=kernel_stack,
            bias=bias_stack,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.conv_groups)
        # Reshape the last channel to create a vector valued RnxH feature map
        output = torch.stack(torch.split(output, self.N_out, 1), 2)

        #kernel_stack = torch.stack([self.kernel(self.h_grid.grid[i]) for i in range(self.N_h)], dim=1)
        # ks = kernel_stack.shape
        # kernel_stack = torch.reshape(kernel_stack, [ks[0] * ks[1], ks[2], ks[-2], ks[-1]])
        # output_2 = torch.conv2d(
        #     input=input,
        #     weight=kernel_stack,
        #     bias=None,
        #     stride=self.stride,
        #     padding=self.padding,
        #     dilation=self.dilation,
        #     groups=self.conv_groups)
        # output_2=output_2.reshape(output_2.shape[0], self.N_out, self.N_h, output_2.shape[-2], output_2.shape[-1])
        # Return the output
        return output


##########################################################################
############################### ConvGGLayer ##############################
##########################################################################
# Start of group_conv class
class ConvGGLayer(ConvRnGLayer, torch.nn.Module):
    def __init__(
            self,
            group,
            N_in,
            N_out,
            kernel_size,
            stride,
            padding,
            dilation,
            conv_groups,
            wscale
            ):
        torch.nn.Module.__init__(self)
        ## Assert and set inputs
        self.kernel_type = 'G'
        self._assert_and_set_inputs(group, N_in, N_out, kernel_size,  stride, padding, dilation, conv_groups)
        ## Construct the trainable weights and initialize them
        weights = []
        self.biases = torch.nn.ParameterList([torch.nn.Parameter(torch.Tensor(self.N_out,  1,1))])
        for i in group.H.rots:
            w = torch.nn.Parameter(torch.Tensor(self.N_out, self.N_in, 1, self.kernel_size, self.kernel_size, self.kernel_size))
            weights.append(w)
        self.weights = torch.nn.ParameterList(weights)
        self._reset_parameters(wscale=wscale)

########################### Assert and set inputs ########################

    # Method overriding:
    def _assert_and_set_inputs(self, group, N_in, N_out, kernel_size, stride, padding, dilation, conv_groups):
        # Default Rn assertions
        self._assert_and_set_inputs_GG(group, N_in, N_out, kernel_size, stride, padding, dilation, conv_groups)
        # Specific initialization/assertion
        self.N_h = group.H.rots  # Target sampling

    def _assert_and_set_inputs_GG(self, group, N_in, N_out, kernel_size, stride, padding, dilation, conv_groups):
        ## Check (and parse) all the inputs
        # Include the dictionary of the used parent class
        self.group = group
        self.H = group.H
        self.Rn = group.Rn
        # Mandatory inputs
        self.N_in = self._assert_N_in(N_in)
        self.N_out = self._assert_N_out(N_out)
        self.kernel_size = self._assert_kernel_size(kernel_size)
        # Optional arguments
        self.conv_groups = self._assert_conv_groups(conv_groups)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

######################### Compute the output ##########################

    # Method overriding:
    def forward(self, input):
        return self.conv_G_G(input)

    # Method overriding:
    def kernel_and_bias(self, h=0):
        # The transformation to apply
        # Sample the kernel on the (transformed) grid
        h_weight, h_bias = self.H.left_representation_on_G(h, self.weights, self.biases)
        return  h_weight, h_bias

    def conv_G_G(self, input):
        # Generate the full stack of convolution kernels (all transformed copies)
        kernel_stack=[]
        bias_stack=[]
        for i in self.N_h:
            w,b = self.kernel_and_bias(h=i)
            kernel_stack.append(w)
            bias_stack.append(b)
            
        kernel_stack = torch.cat(kernel_stack, dim=0) 
        bias_stack = torch.cat(bias_stack, dim=0) 
        


        # Reshape input tensor and kernel as if they were Rn tensors
        kernel_stack_as_if_Rn = torch.reshape(kernel_stack, [len(self.N_h) * self.N_out, self.N_in * len(self.N_h),self.kernel_size, self.kernel_size, self.kernel_size])
        bias_stack_as_if_Rn =torch.squeeze(torch.reshape(bias_stack, [1,len(self.N_h) * self.N_out]))
        input_tensor_as_if_Rn = torch.reshape(input, [input.shape[0], self.N_in * len(self.N_h),input.shape[-3], input.shape[-2], input.shape[-1]])
        # And apply them all at once
        output = torch.conv3d(
            input=input_tensor_as_if_Rn,
            weight=kernel_stack_as_if_Rn,
            bias = bias_stack_as_if_Rn,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.conv_groups)
        # Reshape the last channel to create a vector valued RnxH feature map
        output = torch.stack(torch.split(output, self.N_out, 1), 2)

        # The above includes integration over S1, take discretization into account
        # # Return the output
        return output

