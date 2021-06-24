import torch 
import u_net, eq_net_T4, cnf_net
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

def calc_preds(net_nn, device, input_map_not_padded_3D, Lin=None, Lout=None):
    input_map_3D = np.pad(input_map_not_padded_3D,((pad,pad),(pad,pad),(pad,pad)),'constant', constant_values=0)
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
    out_preds_pad_4D = out_preds_pad_4D[:,pad:-pad,pad:-pad,pad:-pad]
    return out_preds_pad_4D


def calc_confidence(map_file_no_norm_no_pad, pred_map_no_norm_no_pad, conf_net, device):

    map_file_no_norm = np.pad(map_file_no_norm_no_pad,((pad,pad),(pad,pad),(pad,pad)),'constant', constant_values=0)
    pred_map_no_norm = np.pad(pred_map_no_norm_no_pad,((0,0),(pad,pad),(pad,pad),(pad,pad)),'constant', constant_values=0)


    true_false_labels_dummy = map_file_no_norm*0
    submap_dict_img  = maps_to_submaps(map_file_no_norm, true_false_labels_dummy, Lin=LIN, Lout=LOUT)
    submap_dict_pred = maps_to_submaps_4D(pred_map_no_norm, Lin=LIN, Lout=LOUT)

    input_map_3D = map_file_no_norm
    out_labels_pad_4D = np.zeros((N_LABELS, input_map_3D.shape[0],input_map_3D.shape[1], input_map_3D.shape[2]))
    out_preds_pad_4D_CNF = np.zeros((2, input_map_3D.shape[0],input_map_3D.shape[1], input_map_3D.shape[2]))
    out_preds_pad_4D_CNF_TRUE = np.zeros((2, input_map_3D.shape[0],input_map_3D.shape[1], input_map_3D.shape[2]))

    for in_patch, ky in enumerate(list(submap_dict_img.keys())):

        tf_label = submap_dict_img[ky]["label"]
        em_map   = submap_dict_img[ky]["em_map"]
        em_pred  = submap_dict_pred[ky]

        tf_inp = np.concatenate((em_map, em_pred), axis=0)

        xx_5D  = torch.tensor(tf_inp).float().unsqueeze(0).to(device)

        outputs_5D = conf_net(xx_5D)

        out_pred_pad_batch_5D = outputs_5D.cpu().detach().numpy()
        in_x,in_y,in_z = ky
        out_preds_pad_4D_CNF[:,in_x:in_x+LOUT ,in_y:in_y+LOUT,in_z:in_z+LOUT] = out_pred_pad_batch_5D[0,:,:,:,:]
        out_preds_pad_4D_CNF_TRUE[:,in_x:in_x+LOUT ,in_y:in_y+LOUT,in_z:in_z+LOUT] = tf_label
        print(in_patch, len(list(submap_dict_img.keys())), time.ctime())


    if pad != 0:
        out_preds_pad_4D_CNF = out_preds_pad_4D_CNF[:,pad:-pad,pad:-pad,pad:-pad]
        out_preds_pad_4D_CNF_TRUE = out_preds_pad_4D_CNF_TRUE[:,pad:-pad,pad:-pad,pad:-pad]


    
    return out_preds_pad_4D_CNF,out_preds_pad_4D_CNF_TRUE





if __name__ == '__main__':

    ## SELECT DEVICE ##
    if torch.cuda.is_available():
        use_cuda = True
    else:
        use_cuda = False

    device = torch.device("cuda:0" if use_cuda else "cpu")
    print("Device: ",device)


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
    cnf_net = cnf_net.Net("CNF KLein4")
    cnf_net.load_state_dict(torch.load(cnf_net_weights_file, map_location = device))

    
    
    #load and normalize map
    map_file_no_norm = np.load(input_npy_file)
    map_file_no_norm = np.pad(map_file_no_norm,((pad,pad),(pad,pad),(pad,pad)),'constant', constant_values=0)
    # Run SEG-NET
    segm_map_pred_4D = calc_preds(seg_net, device, map_file_no_norm, Lin=LIN, Lout=LOUT)
    print("DEBUG 2131", segm_map_pred_4D.shape)
    cnf_map_pred_4D,_=  calc_confidence(map_file_no_norm, segm_map_pred_4D, cnf_net, device)

    segm_map_labels_3D = np.argmax(segm_map_pred_4D, axis=0)
    cnf_map_labels_3D = np.argmax(cnf_map_pred_4D, axis=0)
    np.save(seg_out_file,segm_map_labels_3D)
    np.save(cnf_out_file,cnf_map_labels_3D)


