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


BATCH_SIZE_TRAIN = 2
BATCH_SIZE_TEST = 8

N_GROUPS = 12

train_maps_list = "/home/disk/res320/iter2/L_320_train.txt"
valid_maps_list = "/home/disk/res320/iter0/L_320_valid.txt"
out_folder = "//home/disk/temp/"
init_weights_file = "/home/disk/res320/iter1/CLF_30000.pth"
time_str = time.ctime().replace(' ','_').replace(':','_')

## SELECT DEVICE ##
if torch.cuda.is_available():
    use_cuda = True
else:
    use_cuda = False

device = torch.device("cuda:0" if use_cuda else "cpu")
print("Device: ",device)
print("GPU COUNT", torch.cuda.device_count()) 


if N_GROUPS == 12:
    out_folder = out_folder + "G12"+"_"+time_str+"/"
    clf_net = eq_net_T4.Net('Segment_Net')
if N_GROUPS == 24:
    out_folder = out_folder + "G24"+"_"+time_str+"/"
    clf_net = eq_net_S4.Net('Segment_Net')
if N_GROUPS == 4:
    out_folder = out_folder + "G4"+"_"+time_str+"/"
    clf_net = eq_net_K4.Net('Segment_Net')    
if N_GROUPS == 1:
    out_folder = out_folder + "G1"+"_"+time_str+"/"
    clf_net = eq_net_Z.Net('Segment_Net')    


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

## CREATE NETWORK
clf_net.to(device)
clf_net.load_state_dict(torch.load(init_weights_file))
clf_net.eval()

all_vars = clf_net.named_parameters()
learnable_pars =[]
for p_name, p_var in all_vars:
    for layer_name in learn_layers:
         if  layer_name in p_name: 
             learnable_pars.append(p_var)


weights = np.ones(22)
weights[0] = 0.05
weights_tensor = torch.tensor(weights).float().to(torch.device(device))

clf_net.to(torch.device(device))
crit_clf = torch.nn.NLLLoss(weight=weights_tensor)
opt_clf = torch.optim.Adam(learnable_pars)

## LOAD DATABASES
train_set = dbloader.PATCHES_DATASET("train ", train_maps_list)
train_loader = torch.utils.data.DataLoader(train_set, batch_size = BATCH_SIZE_TRAIN, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
train_iter = cycle(train_loader)

test_set = dbloader.PATCHES_DATASET("test ", valid_maps_list)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = BATCH_SIZE_TEST, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

res_test = {}
for k in utils.label_dict.keys():
    res_test[k] = {}

# TRAIN CLF-NET
for n_batch_all in range(1000000):

    sample_batched = next(train_iter)

    x = sample_batched['em_map'].float().to(torch.device(device))
    y_true = sample_batched["label"].long().to(torch.device(device))


    #if  n_batch_all%1000==0:
    #    opt_clf.state=collections.defaultdict(dict)

    opt_clf.zero_grad()
    y_clf = clf_net(x)
    loss_clf = crit_clf(y_clf,y_true)
    loss_clf.backward()
    opt_clf.step()

    ## WRITE REPORTS
    if n_batch_all % 10==0:

        acc_clf = utils.acc_batch(y_clf.to('cpu').detach().numpy(), y_true.to('cpu').detach().numpy())
        writer.add_scalars("Loss", {'train CLF ':loss_clf.item()}, n_batch_all)
        writer.add_scalars("Acc",{'train CLF':acc_clf}, n_batch_all)
        writer.flush()
        print("BATCH "  + str( n_batch_all) + "  "+time.ctime())
        print("TRAIN  ", "ACC ", acc_clf, "LOSS :" ,loss_clf.item())


    if n_batch_all % 1000==0:
        loss_clf_test = 0
        acc_clf_test  = 0
        for k in utils.label_dict.keys():
            res_test[k]["tp"]=0
            res_test[k]["fp"]=0
            res_test[k]["fn"]=0
        for i_batch, sample_batched in enumerate(test_loader):

            ts1 = time.time()
            x  = sample_batched['em_map'].float().to(torch.device(device))
            y_true = sample_batched["label"].long().to(torch.device(device))
        
            outputs = clf_net(x)
            loss = crit_clf(outputs, y_true)
            loss_clf_test += loss.item()
            out_np = outputs.to('cpu').detach().numpy()
            y_true_np =  y_true.to('cpu').detach().numpy()
            acc_clf_test += utils.acc_batch(out_np, y_true_np)

            res_batch = utils.acc_by_type_batch(out_np, y_true_np)

            for ky in res_test.keys():
                res_test[ky]["tp"]+=  res_batch[ky]["tp"]
                res_test[ky]["fp"]+=  res_batch[ky]["fp"]
                res_test[ky]["fn"]+=  res_batch[ky]["fn"]

        loss_clf_test = loss_clf_test/i_batch
        acc_clf_test = acc_clf_test/i_batch

        for ky in res_test.keys():
            precision = res_test[ky]["tp"]/(res_test[ky]["tp"] + res_test[ky]["fp"])
            recall  = res_test[ky]["tp"]/(res_test[ky]["tp"] + res_test[ky]["fn"])
            writer.add_scalars(ky, {"Prec":precision, "Rec": recall}, n_batch_all)
        
        print("FINISHED TEST", time.ctime(), " BATCHES ", i_batch)


        writer.add_scalars("Loss", {'test CLF':loss_clf_test}, n_batch_all)
        writer.add_scalars("Acc",{'test CLF':acc_clf_test}, n_batch_all)
        writer.flush()

        print("ACC  ", "TRAIN :", acc_clf, "TEST :" ,acc_clf_test)

        print("LEU ACC ","TP",res_test["LEU"]["tp"] ,"FN",res_test["LEU"]["fn"] ,"FP",res_test["LEU"]["fp"] ,)
        print("BCK ACC ","TP",res_test["NONE"]["tp"] ,"FN",res_test["NONE"]["fn"] ,"FP",res_test["NONE"]["fp"] ,)



    
    if n_batch_all % 5000==0: 
        #save only
        clf_weights_file = nets_folder+ "CLF_" +str(n_batch_all) + str(".pth") 
        torch.save(clf_net.state_dict(), clf_weights_file)
