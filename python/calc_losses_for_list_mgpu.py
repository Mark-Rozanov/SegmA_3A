import torch 
import u_net, eq_net_T4, cnf_net
import time
import numpy as np
import sys
import dbloader

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


BATCH_SIZE_GPU = 6

def calc_losses_clf_net(input_list_file,output_list_file,clf_weight_file, header_to_print = "START"):
    weights = np.ones(22)
    weights[0] = 0.05
    weights_tensor = torch.tensor(weights).float().to("cpu")
    crit_clf = torch.nn.NLLLoss(weight=weights_tensor)


    with open(output_list_file,"w"):
        print(header_to_print)
    print("GPU COUNT", torch.cuda.device_count()) 
    device_ids = list(range(torch.cuda.device_count()))
    batch_size = BATCH_SIZE_GPU*len(device_ids)

    clf_net_replicas = []
    print("DEVICES", device_ids)
    for d in device_ids:
            ## load nets
        clf_net = eq_net_T4.Net("CLF NET T4")
        clf_net.load_state_dict(torch.load(clf_weight_file, map_location = "cpu"))
        d_net = "cuda:{}".format(d)

        clf_net.to(d_net)
        clf_net.eval()
        clf_net_replicas.append(clf_net)

    test_set = dbloader.PATCHES_DATASET("test ", input_list_file)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)


    for i_batch, sample_batched in enumerate(test_loader):
        if i_batch%100 == 0:
            print(i_batch*batch_size,"/",len(test_loader)*batch_size, " PATCHES TESTED", time.ctime())


        y_true_cpu = sample_batched["label"].long().to("cpu")
        x_cpu  = sample_batched['em_map'].float().to("cpu")
        x_scattered  = torch.nn.parallel.scatter(x_cpu, device_ids)
        outputs_scattered = torch.nn.parallel.parallel_apply(clf_net_replicas, x_scattered)
        outputs_cpu = torch.nn.parallel.gather(outputs_scattered, 'cpu')

        with open(output_list_file,"a") as f:
            for i in range(batch_size):
                loss = crit_clf(torch.unsqueeze(outputs_cpu[i,:,:,:,:],0), torch.unsqueeze(y_true_cpu[i,:,:,:],0)).detach().item()
                f.write(sample_batched["file_name"][i])
                f.write(",")
                f.write(str(loss))
                f.write("\n")


if __name__ == '__main__':
    clf_weight_file = "/home/disk/res320/iter2/CLF_30000.pth"

    input_list_file = "//home/disk/res320/iter0/L_320_valid.txt"
    output_list_file = "//home/disk/res320/iter2/L_320_valid_losses_start.txt"
    calc_losses_clf_net(input_list_file,output_list_file,clf_weight_file, header_to_print = "START V320")

    input_list_file = "//home/disk/res320/iter0/L_300_valid.txt"
    output_list_file = "//home/disk/res320/iter2/L_300_valid_losses_start.txt"
    calc_losses_clf_net(input_list_file,output_list_file,clf_weight_file, header_to_print = "START V300")

    input_list_file = "//home/disk/res320/iter2/L_320_cand.txt"
    output_list_file = "//home/disk/res320/iter2/L_320_cand_losses_start.txt"
    calc_losses_clf_net(input_list_file,output_list_file,clf_weight_file, header_to_print = "START CAND 320")

    input_list_file = "//home/disk/res320/iter2/L_320_train.txt"
    output_list_file = "//home/disk/res320/iter2/L_320_train_losses_start.txt"
    calc_losses_clf_net(input_list_file,output_list_file,clf_weight_file, header_to_print = "START train 320")

