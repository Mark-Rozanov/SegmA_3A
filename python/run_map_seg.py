import torch 
import u_net,eq_net_T4
import time
import numpy as np
import sys

pad = 30
BoxIn = 11
BoxOut = 1
HALF_BOX = (BoxIn-BoxOut)//2 
LIN =BoxIn*4
LOUT=LIN - (BoxIn-BoxOut)

N_LABELS=22

MEAN_MAX = 1
MEAN_MIN = 0.05
MEAN = (MEAN_MAX+MEAN_MIN)/2

STD_MIN = 0.05
STD_MAX = 0.3
SIGMA = (STD_MIN+STD_MAX)/2

if __name__ == '__main__':
    input_npy_file = sys.argv[1]
    seg_net_weights_file = sys.argv[2]
    cnf_net_weights_file = sys.argv[3]
    seg_out_file = sys.argv[4]
    cnf_out_file = sys.argv[5]





## SELECT DEVICE ##
if torch.cuda.is_available():
    use_cuda = True
else:
    use_cuda = False

device = torch.device("cuda:0" if use_cuda else "cpu")
print("Device: ",device)

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

    submaps_dict_4D = {}

    for in_x in x:
        for in_y in y:
            for in_z in z:
                submap_3D = np.copy(inp_map[in_x-D:in_x+Lin-D, in_y-D:in_y+Lin-D, in_z-D:in_z+Lin-D])
                sublable_3D = np.copy(inp_label[in_x:in_x+Lout ,in_y:in_y+Lout,in_z:in_z+Lout])
                sun_norm_3D = normalize_3D_box(submap_3D, mean=MEAN, sigma = SIGMA)
                sun_norm_exp_4D = np.expand_dims(sun_norm_3D,0)
                submaps_dict_4D[(in_x,in_y,in_z)] = {"em_map":sun_norm_exp_4D, "label":sublable_3D}


    return submaps_dict_4D

def calc_preds(net_nn, device, input_map_not_padded_3D, Lin=None, Lout=None):
    input_map_3D = np.pad(input_map_not_padded_3D,((Lin,Lin),(Lin,Lin),(Lin,Lin)),'constant', constant_values=0)
    out_preds_pad_4D = np.zeros((N_LABELS, input_map_3D.shape[0],input_map_3D.shape[1], input_map_3D.shape[2]))

    submaps_dict = maps_to_submaps(input_map_3D, out_preds_pad_4D, Lin=Lin, Lout=Lout)
    for in_key, ky in enumerate(list(submaps_dict.keys())):
        xx_5D  = torch.tensor(submaps_dict[ky]['em_map']).float().unsqueeze(0).to(device)

        outputs_5D = net_nn(xx_5D)

        out_pred_pad_batch_5D = outputs_5D.cpu().detach().numpy()
        in_x,in_y,in_z = ky
        out_preds_pad_4D[:,in_x:in_x+Lout ,in_y:in_y+Lout,in_z:in_z+Lout] = out_pred_pad_batch_5D[0,:,:,:,:]
        print(in_key, len(list(submaps_dict.keys())), time.ctime())

    Lin = np.int(Lin)
    out_preds_pad_4D = out_preds_pad_4D[:,Lin:-Lin,Lin:-Lin,Lin:-Lin]
    return out_preds_pad_4D


if __name__ == '__main__':
    input_npy_file = sys.argv[1]
    seg_net_weights_file = sys.argv[2]
    cnf_net_weights_file = sys.argv[3]
    seg_out_file = sys.argv[4]
    cnf_out_file = sys.argv[5]

    ## load nets
    clf_net = eq_net_T4.Net("CLF NET T4")
    seg_net = u_net.Net('Segment_Net', clf_net=clf_net)
    seg_net.to(device)
    seg_net.load_state_dict(torch.load(seg_net_weights_file, map_location = device))
    seg_net.eval()
    #load and normalize map
    map_file_no_norm = np.load(input_npy_file)
    map_file_no_norm = np.pad(map_file_no_norm,((pad,pad),(pad,pad),(pad,pad)),'constant', constant_values=0)
    # Run SEG-NET
    segm_map_no_norm = calc_preds(seg_net, device, map_file_no_norm, Lin=LIN, Lout=LOUT)

