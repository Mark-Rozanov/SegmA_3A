import torch
import eq_net_T4, seg_net, cnf_net, tests_segma
import time
import numpy as np
import sys
import os,shutil

if __name__ == '__main__':

    ## SELECT DEVICE ##
    if torch.cuda.is_available():
        use_cuda = True
    else:
        use_cuda = False

    device = torch.device("cuda:0" if use_cuda else "cpu")
    print("Device: ",device)
    print("GPU COUNT", torch.cuda.device_count())



    map_folder = sys.argv[1]
    cnf_net_weights_file = sys.argv[2]

    ## SEGMENTATION AND CONFIDENCE
    clf_nn = eq_net_T4.Net("CLF NET T4")
    seg_nn = seg_net.Net('Segment_Net', clf_net=clf_nn)
    cnf_nn = cnf_net.Net('CNF Net', seg_nn = seg_nn)
    cnf_nn.load_state_dict(torch.load(cnf_net_weights_file))
    cnf_nn.eval()
    cnf_nn.to(device)

    res_folder = map_folder+'/res/'
    if  os.path.exists(res_folder):
        shutil.rmtree(res_folder)
    os.mkdir(res_folder)

    tests_segma.run_test_on_whole_map(cnf_nn = cnf_nn, in_map_fold=map_folder, out_map_fold=res_folder,device_id=device)
