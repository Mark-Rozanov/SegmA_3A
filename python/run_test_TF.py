import os,sys,shutil
sys.path.append("/home/iscb/wolfson/Mark/git/work_from_home/AASegm/python")
import numpy as np

import dbloader
import  u_net, tf_eq_net
import torch

u_net_weights_file = "//home/iscb/wolfson/Mark/data/AASegTorch/out_TF_2_Jan31/net_TF99000.pth"
tf_net_weights_file = "//home/iscb/wolfson/Mark/data/AASegTorch/out_UN_tf/net_TF24000.pth"


test_maps = ["emd_11221_rot2", "emd_11993_rot3", "emd_12130_rot4","emd_30639_rot4"]

work_folder = "/home/iscb/wolfson/Mark/data/AASegTorch/results/"
ref_folder = "/home/iscb/wolfson/Mark/data/AASegTorch/results/ref_data/"


seg_net = u_net.Net('Segment_Net')
seg_net.to(torch.device("cuda:0"))
seg_net.load_state_dict(torch.load(u_net_weights_file))
seg_net.eval()

tf_net = tf_eq_net.Net('tf_net')
tf_net.to(torch.device("cuda:0"))
tf_net.load_state_dict(torch.load(tf_net_weights_file))
tf_net.eval()


for t_map in test_maps:
    #create folder and files
    res_folder = work_folder + "//" + t_map + "//"
    shutil.rmtree(res_folder, ignore_errors=True)
    os.mkdir(res_folder)

    unput_file_source = ref_folder + t_map + "_map.npy"
    true_file_source = ref_folder + t_map + "_label.npy"

    unput_file = res_folder  + "input_map.npy"
    true_file = res_folder  + "input_label.npy"

    shutil.copyfile(unput_file_source, unput_file)
    shutil.copyfile(true_file_source, true_file)


    net_tf_res_file = res_folder +"map_tf.npy"
    net_seg_res_file = res_folder +"map_seg.npy"

 
    ## run tests
    Lin=44
    Lout =34
    input_matrix = np.load(unput_file)
    test_dataset=dbloader.TF_DATASET_ONE("input", seg_net, res_folder,non_thr = -1, pad = 50)
    
    seg_labels , tf_labels , tf_labels_true = test_dataset.calc_test_tf(tf_net)


    np.save(net_tf_res_file, tf_labels)
    np.save(net_seg_res_file, seg_labels)
