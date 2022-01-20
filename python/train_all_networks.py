
import copy
import os,sys

# Import the group structure
import torch
import torch.nn as nn
import numpy as np
from itertools import cycle
import torch.optim as optim
import copy
import torchvision
#from torch.utils.tensorboard import SummaryWriter
import shutil
import eq_net_T4, seg_net, cnf_net, atoms_net
import glob
#from dbloader import VX_BOX_SIZE, MAP_BOX_SIZE, N_CHANNELS, BATCH_SIZE
import utils
#from dbloader import  MEAN_MIN, MEAN_MAX, STD_MIN, STD_MAX
from torch.utils.data import Dataset, DataLoader
import importlib
import time
import collections
import re
import math

import dbloader, work_with_lists, protein_atoms, tests_segma
# In[2]:

N_GROUPS_CLF=12
N_GROUPS_CNF=24

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
BATCH_SIZE_TRAIN = 2
BATCH_SIZE_TEST = 8
N_EPOCHS = 4

from utils import Atom, element_dict, write_atoms_file, write_pdb_file



def get_train_data_loader(db_folder, train_pdbs, batch_size=None, net_thr=None):
    one_map_datasets=[]
    for p in train_pdbs:
        one_map_datasets.append(EM_DATASET_ONE(p,db_folder, non_thr = net_thr))

    ln = [len(db) for db in one_map_datasets ]
    in_ad = np.round(np.max(ln)/np.array(ln)).astype(int)
    balanced_datasets = []
    for i in range(len(in_ad)):
        for ii in range(in_ad[i]):
            balanced_datasets.append(one_map_datasets[i])
    train_dataset =  torch.utils.data.ConcatDataset(balanced_datasets)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    return train_loader, len(train_loader)



def trainSEG(train_maps_list,valid_maps_list,root_folder, max_batches = 1000000 ):

    device_ids = list(range(torch.cuda.device_count()))

    device_seg = 'cuda:{}'.format(device_ids[0])

    clf_nn = eq_net_T4.Net('CLF NET')
    seg_nn = seg_net.Net('Seg Net', clf_net = clf_nn)


    seg_weights_file = root_folder +"/seg_net.pth"
    clf_weights_file = root_folder +"/clf_net.pth"

#    seg_nn.load_state_dict(torch.load(seg_weights_file))
#    seg_nn.eval()
    clf_nn.load_state_dict(torch.load(clf_weights_file))
    clf_nn.eval()
    seg_nn.to(device_seg)

    ### make out folders
    time_str = time.ctime().replace(' ','_').replace(':','_')
    seg_out_folder = root_folder + "/SEG"+"_"+time_str+"/"
    seg_graphs_folder = seg_out_folder+'/graphs/'
    seg_nets_folder = seg_out_folder+'/nets/'
    seg_res_folder = seg_out_folder+'/results/'
    os.mkdir(seg_out_folder)
    os.mkdir(seg_graphs_folder)
    os.mkdir(seg_nets_folder)
    os.mkdir(seg_res_folder)

    seg_writer =  SummaryWriter(log_dir=seg_graphs_folder)

    weights_seg = np.ones(22)
    weights_seg[0] = 0.01
    weights_seg[21] = 10

    weights_tensor_seg = torch.tensor(weights_seg).float().cuda(device_seg)
    crit_seg = torch.nn.CrossEntropyLoss(weight=weights_tensor_seg)

    seg_nn_vars = seg_nn.get_trained_parameters()
    opt_seg = torch.optim.Adam(seg_nn_vars,amsgrad=True,  weight_decay=0.0000001)

    ## LOAD DATABASES
    train_set = dbloader.PATCHES_DATASET("train ", train_maps_list)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = BATCH_SIZE_TRAIN, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    train_iter = cycle(train_loader)

    test_set = dbloader.PATCHES_DATASET("test ", valid_maps_list)

    all_batches = int(len(train_loader)*N_EPOCHS)
    all_batches = np.minimum(max_batches,all_batches)
    # TRAIN CLF-NET
    for n_batch_all in range(all_batches):

        sample_batched = next(train_iter)

        #GET DATA
        x  = sample_batched['em_map'].float()
        labels_true = sample_batched["label"].long()

        #SEGMENTATIONs
        opt_seg.zero_grad()
        y_seg, y_clf = seg_nn(x.cuda(device_seg))
        loss_seg = crit_seg(y_seg,labels_true.cuda(device_seg))

        loss_seg.backward(retain_graph=True)
        opt_seg.step()

        if n_batch_all== all_batches//4:
            weights_seg = np.ones(22)
            weights_seg[0] = 0.05
            weights_seg[21] = 10

            weights_tensor_seg = torch.tensor(weights_seg).float().cuda(device_seg)
            crit_seg = torch.nn.CrossEntropyLoss(weight=weights_tensor_seg)
            opt_seg = torch.optim.Adam(seg_nn_vars,amsgrad=True)

        ## WRITE REPORTS
        ## WRITE REPORTS
        if n_batch_all % 200==0 :
            test_batches = 20
            if n_batch_all % 1000== 0 :
                test_batches = 50

            acc_seg = utils.acc_batch(y_seg.to('cpu').detach().numpy(), labels_true.to('cpu').detach().numpy())
            seg_writer.add_scalars("Loss", {'train SEG ':loss_seg.item()}, n_batch_all)
            seg_writer.add_scalars("Acc",{'train SEG':acc_seg}, n_batch_all)
            seg_writer.flush()

            print("BATCH "  + str( n_batch_all) +"/"+ str(all_batches)+ "  "+time.ctime())
            print("TRAIN  ", "SEG ACC ", acc_seg, "LOSS SEG :" ,loss_seg.item())

            seg_acc, seg_loss = tests_segma.write_labeling_results_test(test_set = test_set, out_name='label', device_id = device_seg,\
            train_batch_num = n_batch_all, max_test_batches = test_batches,\
             lbl_func=clf_nn, lbl_crit=crit_seg, writer=seg_writer, l_dict=utils.label_dict,label_suffix="_clf")


            seg_acc, seg_loss = tests_segma.write_labeling_results_test(test_set = test_set, out_name='label', device_id = device_seg,\
            train_batch_num = n_batch_all, max_test_batches = test_batches,\
             lbl_func=seg_nn, lbl_crit=crit_seg, writer=seg_writer, l_dict=utils.label_dict)
            print("TEST  SEG  ACC: ",seg_acc, " LOSS: ", seg_loss)



        if n_batch_all % 1000==0  :
            #save only

            seg_weights_file = seg_nets_folder+ "SEG_" +str(n_batch_all) + str(".pth")
            last_weights_file = root_folder + '/seg_net.pth'
            torch.save(seg_nn.state_dict(), seg_weights_file)
            torch.save(seg_nn.state_dict(), last_weights_file)


def trainCLF(train_maps_list,valid_maps_list,root_folder, max_batches = 1000000 ):

    device_ids = list(range(torch.cuda.device_count()))
    device_clf = 'cuda:{}'.format(device_ids[0])

    clf_nn = eq_net_T4.Net('CLF NET')

    weights_file = root_folder +"/clf_net.pth"

    clf_nn.load_state_dict(torch.load(weights_file))
    clf_nn.eval()
    clf_nn.to(device_clf)

    ### make out folders
    time_str = time.ctime().replace(' ','_').replace(':','_')
    out_folder = root_folder + "/CLF"+"_"+time_str+"/"
    graphs_folder = out_folder+'/graphs/'
    nets_folder = out_folder+'/nets/'
    res_folder = out_folder+'/results/'
    os.mkdir(out_folder)
    os.mkdir(graphs_folder)
    os.mkdir(nets_folder)
    os.mkdir(res_folder)
    writer =  SummaryWriter(log_dir=graphs_folder)

    weights = np.ones(22)
    weights[0] = 0.05
    weights[21] = 10

    weights_tensor = torch.tensor(weights).float().cuda(device_clf)
    crit = torch.nn.CrossEntropyLoss(weight=weights_tensor)

    nn_vars = clf_nn.parameters()
    opt = torch.optim.Adam(nn_vars,amsgrad=True,weight_decay=0.0000001)

    ## LOAD DATABASES
    train_set = dbloader.PATCHES_DATASET("train ", train_maps_list)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = BATCH_SIZE_TRAIN, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    train_iter = cycle(train_loader)

    test_set = dbloader.PATCHES_DATASET("test ", valid_maps_list)

    all_batches = int(len(train_loader)*N_EPOCHS)
    all_batches = np.minimum(max_batches,all_batches)
    # TRAIN CLF-NET
    for n_batch_all in range(all_batches):

        sample_batched = next(train_iter)

        #GET DATA
        x  = sample_batched['em_map'].float()
        labels_true = sample_batched["label"].long()

        #SEGMENTATIONs
        opt.zero_grad()
        y = clf_nn(x.cuda(device_clf))
        loss_clf = crit(y,labels_true.cuda(device_clf))

        loss_clf.backward(retain_graph=True)
        opt.step()

        if n_batch_all== all_batches//2:
            weights = np.ones(22)
            weights[0] = 0.05
            weights[21] = 10

            weights_tensor = torch.tensor(weights).float().cuda(device_clf)
            crit = torch.nn.CrossEntropyLoss(weight=weights_tensor)

        ## WRITE REPORTS
        if n_batch_all % 200==0 :
            test_batches = 10
            if n_batch_all % 1000== 0 :
                test_batches = 100

            acc = utils.acc_batch(y.to('cpu').detach().numpy(), labels_true.to('cpu').detach().numpy())
            writer.add_scalars("Loss", {'train':loss_clf.item()}, n_batch_all)
            writer.add_scalars("Acc",{'train':acc}, n_batch_all)
            writer.flush()

            print("BATCH "  + str( n_batch_all) +"/"+ str(all_batches)+ "  "+time.ctime())
            print("TRAIN  ", "CLF ACC ", acc, "LOSS CLF :" ,loss_clf.item())

            clf_acc, clf_loss = tests_segma.write_labeling_results_test(test_set = test_set, out_name='label', device_id = device_clf,\
            train_batch_num = n_batch_all, max_test_batches = test_batches,\
            lbl_func=clf_nn, lbl_crit=crit, writer=writer, l_dict=utils.label_dict)
            print("TEST  CLF  ACC: ",clf_acc, " LOSS: ", clf_loss, time.ctime())

        if n_batch_all % 1000==0  :
            #save only
            clf_weights_file = nets_folder+ "CLF_" +str(n_batch_all) + str(".pth")
            last_weights_file = root_folder + '/clf_net.pth'
            torch.save(clf_nn.state_dict(), clf_weights_file)
            torch.save(clf_nn.state_dict(), last_weights_file)


def trainCNF(train_maps_list,valid_maps_list,root_folder, max_batches = 1000000 ):

    device_ids = list(range(torch.cuda.device_count()))
    device = 'cuda:{}'.format(device_ids[0])

    clf_nn = eq_net_T4.Net('CLF NET')
    seg_nn = seg_net.Net('Seg Net', clf_net = clf_nn)

    seg_weights_file = root_folder +"/seg_net.pth"
    seg_nn.load_state_dict(torch.load(seg_weights_file))
    seg_nn.eval()

    cnf_nn = cnf_net.Net('CNF Net', seg_nn = seg_nn)
    cnf_nn.to(device)

    ### make out folders
    time_str = time.ctime().replace(' ','_').replace(':','_')
    out_folder = root_folder + "/CNF"+"_"+time_str+"/"
    graphs_folder = out_folder+'/graphs/'
    nets_folder = out_folder+'/nets/'
    res_folder = out_folder+'/results/'
    os.mkdir(out_folder)
    os.mkdir(graphs_folder)
    os.mkdir(nets_folder)
    os.mkdir(res_folder)
    writer =  SummaryWriter(log_dir=graphs_folder)

    crit = torch.nn.CrossEntropyLoss(reduction='none')
    background_weight = 0.05

    opt = torch.optim.Adam(cnf_nn.get_trained_parameters(),amsgrad=True,weight_decay=0.0000001)

    ## LOAD DATABASES
    train_set = dbloader.PATCHES_DATASET("train ", train_maps_list)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = BATCH_SIZE_TRAIN, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    train_iter = cycle(train_loader)

    test_set = dbloader.PATCHES_DATASET("test ", valid_maps_list)

    all_batches = np.int(len(train_loader)*N_EPOCHS)
    all_batches = np.minimum(max_batches,all_batches)
    # TRAIN CLF-NET
    for n_batch_all in range(all_batches):

        sample_batched = next(train_iter)

        #GET DATA
        x  = sample_batched['em_map'].float()
        labels_true = sample_batched["label"].long()

        #
        opt.zero_grad()

        y_cnf, y_seg_pred, y_clf_pred = cnf_nn(x.cuda(device))
        tf_true_labels = tests_segma.y_true_cnf(y_seg_pred,labels_true.cuda(device))

        loss_per_sample = crit(y_cnf,tf_true_labels)
        sample_weight = torch.where(labels_true==0, background_weight, 1.0).cuda(device)
        loss =(loss_per_sample * sample_weight / sample_weight.sum()).sum()

        loss.backward()
        opt.step()

        ## WRITE REPORTS
        if n_batch_all % 200==0 :
            test_batches = 20
            if n_batch_all % 1000== 0 :
                test_batches = 200


            acc_cnf = utils.acc_batch(y_cnf.to('cpu').detach().numpy(), tf_true_labels.to('cpu').detach().numpy())
            writer.add_scalars("Loss", {'train CNF ':loss.item()}, n_batch_all)
            writer.add_scalars("Acc",{'train CNF':acc_cnf}, n_batch_all)
            writer.flush()

            print("BATCH "  + str( n_batch_all) +"/"+ str(all_batches)+ "  "+time.ctime())
            print("TRAIN  ", "CNF ACC ", acc_cnf, "LOSS CNF :" ,loss.item())

            cnf_acc, cnf_loss = tests_segma.write_labeling_results_test_cnf(test_set = test_set, device_id= device,
            cnf_nn = cnf_nn, train_batch_num = n_batch_all, background_weight = background_weight,\
            max_test_batches = test_batches, lbl_crit=crit, writer=writer)

            print("TEST  CNF  ACC: ",cnf_acc, " LOSS: ", cnf_loss)

        if n_batch_all % 1000==0  :
            weights_file = nets_folder+ "CNF_" +str(n_batch_all) + str(".pth")
            last_weights_file = root_folder + '/cnf_net.pth'
            torch.save(cnf_nn.state_dict(), weights_file)
            torch.save(cnf_nn.state_dict(), last_weights_file)
