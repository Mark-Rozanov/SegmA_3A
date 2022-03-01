import os,sys,shutil
import numpy as np

import dbloader
import torch
import eq_net_T4, eq_net_S4, eq_net_K4, eq_net_Z
import time
import utils
from itertools import cycle
import collections
from torch.utils.tensorboard import SummaryWriter
from utils import acc_batch
import tests_segma
import work_with_lists, calc_losses_for_list_mgpu


BATCH_SIZE_TEST = 8

N_GROUPS = 12
N_EPOCHS = 3
DB_THR = 0.75
BATCH_SIZE_GPU = 6


def calc_losses_clf_net_mgpu(test_set, device_ids, clf_weight_file, crit_clf, header_to_print, output_list_file):
    batch_size = BATCH_SIZE_GPU*len(device_ids)

    with open(output_list_file,"w") as f:
        print("NEW LIST FILE", output_list_file)


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

    test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)


    for i_batch, sample_batched in enumerate(test_loader):
        if i_batch%100== 0:
            print(i_batch*batch_size,"/",len(test_loader)*batch_size, " PATCHES TESTED", time.ctime())


        y_true_cpu = sample_batched["label"].long().to("cpu")
        x_cpu  = sample_batched['em_map'].float().to("cpu")
        x_scattered  = torch.nn.parallel.scatter(x_cpu, device_ids)
        outputs_scattered = torch.nn.parallel.parallel_apply(clf_net_replicas, x_scattered)
        outputs_cpu = torch.nn.parallel.gather(outputs_scattered, device_ids[0]).to('cpu')

        with open(output_list_file,"a") as f:
            for i in range(batch_size):
                loss = crit_clf(torch.unsqueeze(outputs_cpu[i,:,:,:,:],0), torch.unsqueeze(y_true_cpu[i,:,:,:],0)).detach().item()
                f.write(sample_batched["file_name"][i])
                f.write(",")
                f.write(str(loss))
                f.write("\n")


def calc_losses_from_list(test_set, device_id, clf_net, crit_clf, header_to_print, out_list_file):
    with torch.no_grad():
        test_loader = torch.utils.data.DataLoader(test_set, batch_size = BATCH_SIZE_TEST, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)

        for i_batch, sample_batched in enumerate(test_loader):
            if i_batch%100== 0:
                print(i_batch*BATCH_SIZE_TEST,"/",len(test_loader)*BATCH_SIZE_TEST, " PATCHES TESTED", time.ctime())


            y_true = sample_batched["label"].long().to(device_id)
            x  = sample_batched['em_map'].float().to(device_id)
            outputs = clf_net(x)

            with open(out_list_file,"a") as f:
                for i in range(BATCH_SIZE_TEST):
                    loss = crit_clf(torch.unsqueeze(outputs[i,:,:,:,:],0), torch.unsqueeze(y_true[i,:,:,:],0)).detach().item()
                    f.write(sample_batched["file_name"][i])
                    f.write(",")
                    f.write(str(loss))
                    f.write("\n")

    return

def calc_threshold():
    return


def transfer_clf_net(cand_list,valid_maps_list,root_folder,init_weights_file, max_batches = 100000):

    time_str = time.ctime().replace(' ','_').replace(':','_')
    device = torch.device("cuda:0")

    loss_thr = -999

    if N_GROUPS == 12:
        out_folder = root_folder + "/G12"+"_"+time_str+"/"
        clf_nn_train = eq_net_T4.Net('Train')
        clf_nn_ref = eq_net_T4.Net('Reference')

    else:
        error


    learn_layers = ['conv1g','bn1','conv2g','bn2','conv6','bn6']

    ### make out folders
    graphs_folder = out_folder+'/graphs/'
    writer =  SummaryWriter(log_dir=graphs_folder)

    nets_folder = out_folder+'/nets/'
    res_folder = out_folder+'/results/'

    shutil.rmtree(out_folder, ignore_errors=True)
    os.mkdir(out_folder)
    os.mkdir(graphs_folder)
    os.mkdir(nets_folder)
    os.mkdir(res_folder)

    ## CREATE NETWORKS
    clf_nn_train.to(device)
    clf_nn_train.load_state_dict(torch.load(init_weights_file))
    clf_nn_train.eval()

    clf_nn_ref.to(device)
    clf_nn_ref.load_state_dict(torch.load(init_weights_file))
    clf_nn_ref.eval()

    all_vars = clf_nn_train.named_parameters()
    learnable_pars =[]
    for p_name, p_var in all_vars:
        for layer_name in learn_layers:
            if  layer_name in p_name:
                learnable_pars.append(p_var)


    weights = np.ones(22)
    weights[0] = 0.05
    weights[21] = 10
    weights_tensor = torch.tensor(weights).float().to(torch.device(device))

    crit_clf = torch.nn.CrossEntropyLoss(weight=weights_tensor)
    opt_clf = torch.optim.Adam(learnable_pars)

    ## LOAD DATABASES
    train_set = dbloader.PATCHES_DATASET("train ", cand_list)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = 1, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    train_iter = cycle(train_loader)

    test_set = dbloader.PATCHES_DATASET("test ", valid_maps_list)

    res_test = {}
    for k in utils.label_dict.keys():
        res_test[k] = {}

    all_batches = int(len(train_loader)*N_EPOCHS)

    all_batches = np.minimum(all_batches, max_batches)
    # TRAIN CLF-NET
    n_good = 0
    n_bad = 0
    for n_batch_all in range(all_batches):

        sample_batched = next(train_iter)

        x  = sample_batched['em_map'].float().to(torch.device(device))
        y_true = sample_batched["label"].long().to(torch.device(device))

        with  torch.no_grad():
            y_clf = clf_nn_ref(x)
            loss_clf = crit_clf(y_clf,y_true)

        if loss_clf.item()<loss_thr:
            n_good+=1.0

            opt_clf.zero_grad()
            y_clf = clf_nn_train(x)
            loss_clf = crit_clf(y_clf,y_true)
            loss_clf.backward()
            opt_clf.step()
        else:
            n_bad+=1.0

        ## WRITE REPORTS
        ## WRITE REPORTS
        if n_batch_all % 200==0 :
            test_batches = 10
            if n_batch_all % 1000== 0 :
                test_batches = 100

            writer.add_scalars("N samples",{'good':n_good/(n_good+n_bad+0.001),'bad':n_bad/(n_good+n_bad+0.001)}, n_batch_all)
            writer.flush()

            acc = utils.acc_batch(y_clf.to('cpu').detach().numpy(), y_true.to('cpu').detach().numpy())
            writer.add_scalars("Loss", {'train':loss_clf.item(),'thr':loss_thr}, n_batch_all)
            writer.add_scalars("Acc",{'train':acc}, n_batch_all)
            writer.flush()

            print("BATCH "  + str( n_batch_all) +"/"+ str(all_batches)+ "  "+time.ctime())
            print("TRAIN  ", "CLF ACC ", acc, "LOSS CLF :" ,loss_clf.item())

            clf_acc, clf_loss = tests_segma.write_labeling_results_test(test_set = test_set, out_name='label', device_id = device,\
            train_batch_num = n_batch_all, max_test_batches = test_batches,\
            lbl_func=clf_nn_train, lbl_crit=crit_clf, writer=writer, l_dict=utils.label_dict)
            print("TEST  CLF  ACC: ",clf_acc, " LOSS: ", clf_loss, time.ctime())

        if n_batch_all % 10000==0:
            opt_clf = torch.optim.Adam(learnable_pars)

            #save only
            clf_weights_file = nets_folder+ "CLF_" +str(n_batch_all) + str(".pth")
            last_weights_file = root_folder + '/clf_net.pth'
            torch.save(clf_nn_train.state_dict(), clf_weights_file)
            torch.save(clf_nn_train.state_dict(), last_weights_file)
            #update  ref net
            clf_nn_ref.load_state_dict(torch.load(clf_weights_file))
            clf_nn_ref.eval()
            #update
            valid_loss_file = res_folder + "valid_loss_" +str(n_batch_all) + str(".txt")
            calc_losses_from_list(test_set= test_set, device_id=device, clf_net=clf_nn_ref, crit_clf=crit_clf, header_to_print='VALID{}'.format(n_batch_all), out_list_file = valid_loss_file)
            loss_dict_valid = work_with_lists.load_res_file(valid_loss_file)
            loss_thr = np.quantile(loss_dict_valid["loss"],DB_THR)
