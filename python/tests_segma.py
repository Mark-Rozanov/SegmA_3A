import os,sys,shutil
import numpy as np

import dbloader
import torch
import eq_net_T4,seg_net, cnf_net, atoms_net
import time
import utils
from itertools import cycle
import collections
from torch.utils.tensorboard import SummaryWriter
from utils import acc_batch, Atom, write_atoms_file, write_pdb_file
import work_with_lists
import protein_atoms

BATCH_SIZE_TRAIN = 2
BATCH_SIZE_TEST = 8

N_GROUPS = 12
N_EPOCHS = 3

def run_test_on_whole_map(cnf_nn = None, in_map_fold=None, out_map_fold=None, device_id = None):
    out_atoms_file = out_map_fold +'/atoms_out.txt'
    out_pdbs_file = out_map_fold +'/prot_out.pdb'
    seg_labels_file = out_map_fold +'/seg_labels.npy'
    true_labels_file = out_map_fold +'/true_labels.npy'
    clf_labels_file = out_map_fold +'/clf_labels.npy'
    out_tf_file = out_map_fold +'/tf_out.npy'
    atoms_map_file = out_map_fold +'/atoms_map_out.npy'
    out_em_map_file = out_map_fold +'/em_map.npy'

    # create list
    patches_files  = [in_map_fold + '/test_patches/'+f for f in os.listdir(in_map_fold + '/test_patches/') if f[-10:]=='_label.npy']
    test_list_file = out_map_fold +"/test_patches.txt"
    with open(test_list_file,'w') as f_test:
        for f_name in patches_files:
            f_test.write(f_name)
            f_test.write("\n")

    #create dataset
    test_set = dbloader.PATCHES_DATASET("test ", test_list_file)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = 1, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)

    #get map size
    in_em_map_file = in_map_fold +'/input_map.npy'
    input_map = np.load(in_em_map_file)
    map_size_X, map_size_Y, map_size_Z = input_map.shape

    N_atoms = len(utils.element_dict.keys())

    seg_labels = np.zeros((22,map_size_X,map_size_Y,map_size_Z))
    clf_labels = np.zeros((22,map_size_X,map_size_Y,map_size_Z))
    cnf_labels = np.zeros((2,map_size_X,map_size_Y,map_size_Z))
    true_labels = np.zeros((map_size_X,map_size_Y,map_size_Z))
    out_tf = np.zeros((2,map_size_X,map_size_Y,map_size_Z))
    out_atoms = np.zeros((N_atoms,map_size_X,map_size_Y,map_size_Z))
    true_atoms = np.zeros((map_size_X,map_size_Y,map_size_Z))

    for i_batch, sample_batched in enumerate(test_loader):

        inp_map  = sample_batched['em_map'].float().to(device_id)
        inp_labels = sample_batched["label"].long().to(device_id)


        in_x, in_y, in_z = sample_batched['corner_pos']

        y_cnf,y_seg,y_clf = cnf_nn(inp_map)
#        y_atoms = atoms_nn(inp_map)

        Lout = y_seg.shape[-1]

        seg_labels[:,in_x:in_x+Lout ,in_y:in_y+Lout,in_z:in_z+Lout] = y_seg[0,:,:,:,:].cpu().detach().numpy()
        clf_labels[:,in_x:in_x+Lout ,in_y:in_y+Lout,in_z:in_z+Lout] = y_clf[0,:,:,:,:].cpu().detach().numpy()
        cnf_labels[:,in_x:in_x+Lout ,in_y:in_y+Lout,in_z:in_z+Lout] = y_cnf[0,:,:,:,:].cpu().detach().numpy()
        true_labels[in_x:in_x+Lout ,in_y:in_y+Lout,in_z:in_z+Lout] = inp_labels[0,:,:,:].cpu().detach().numpy()

    #SAVE MAPS
    seg_labels_file = out_map_fold +'/seg_labels.npy'
    true_labels_file = out_map_fold +'/true_labels.npy'
    clf_labels_file = out_map_fold +'/clf_labels.npy'
    cnf_labels_file = out_map_fold +'/cnf_labels.npy'
#    atoms_map_file = out_map_fold +'/atoms_map_out.npy'
#    true_atoms_file = out_map_fold +'/atoms_true.npy'


    np.save(true_labels_file, true_labels)
    np.save(seg_labels_file, seg_labels)
    np.save(cnf_labels_file, cnf_labels)
    np.save(clf_labels_file, clf_labels)
    np.save(out_em_map_file, input_map)


#    #Calculate Atom Positions
#    atoms_list = []
#    for atom_type in utils.element_dict.keys():
#        if atom_type == "None":
#            continue
#
#        atoms_centers, cent_vals= protein_atoms.get_peaks(out_atoms[utils.element_dict[atom_type],:,:,:])
#        for at in atoms_centers:
#            x,y,z = at
#            new_atom = Atom(atom_type = atom_type,res_type ='ALA', x=x, y=y, z=z)
#            new_atom.label_probs = seg_labels[:,x,y,z].copy()
#            new_atom.true_prob = out_tf[1,x,y,z]
#
#            atoms_list.append(new_atom)
#
#    out_pdbs_file = out_map_fold +'/prot_out.pdb'
#    out_atoms_file = out_map_fold +'/atoms_out.txt'

#    write_atoms_file(out_atoms_file,atoms_list)
#    write_pdb_file(out_pdbs_file,atoms_list)


def write_labeling_results_test(test_set = None, out_name=None, device_id= None, train_batch_num = None,\
max_test_batches = 500, lbl_func=None, lbl_crit=None, writer=None, l_dict=None, label_suffix=""):
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = BATCH_SIZE_TEST, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    test_cycle = cycle(test_loader)

    torch.cuda.empty_cache()

    res_test={}

    loss_test = 0
    acc_test  = 0
    for k in l_dict.keys():
        res_test[k] = {}
        res_test[k]["tp"]=0
        res_test[k]["fp"]=0
        res_test[k]["fn"]=0

    for i_batch in range(min(len(test_loader),max_test_batches )):

        sample_batched = next(test_cycle)

        with torch.no_grad():
            x  = sample_batched['em_map'].float().cuda(device_id)
            y_true = sample_batched[out_name].long().cuda(device_id)

            outputs = lbl_func(x)
            if type(outputs) is tuple:
                outputs = outputs[0]

            loss = lbl_crit(outputs, y_true)

        loss_test += loss.item()

        out_np = outputs.to('cpu').detach().numpy()
        y_true_np =  y_true.to('cpu').detach().numpy()
        acc_test += utils.acc_batch(out_np, y_true_np)

        res_batch = utils.acc_by_type_batch(out_np, y_true_np,l_dict)

        for ky in res_test.keys():
            res_test[ky]["tp"]+=  res_batch[ky]["tp"]
            res_test[ky]["fp"]+=  res_batch[ky]["fp"]
            res_test[ky]["fn"]+=  res_batch[ky]["fn"]

    loss_test = loss_test/i_batch
    acc_test = acc_test/i_batch

    for ky in res_test.keys():
        precision = res_test[ky]["tp"]/(res_test[ky]["tp"] + res_test[ky]["fp"]+0.001)
        recall  = res_test[ky]["tp"]/(res_test[ky]["tp"] + res_test[ky]["fn"]+0.001)
        writer.add_scalars(ky, {"Prec"+label_suffix : precision, "Rec"+label_suffix : recall}, train_batch_num)

    writer.add_scalars("Loss", {'test'+label_suffix : loss_test}, train_batch_num)
    writer.add_scalars("Acc",{'test'+label_suffix : acc_test}, train_batch_num)
    writer.flush()

    return acc_test, loss_test

def calc_test_lost_acc(nn_net,criterion,test_loader):
    n = 0
    loss_total = 0
    acc_total  = 0
    for i_batch, sample_batched in enumerate(test_loader):
        n+=1
        x  = sample_batched['em_map'].float().to(torch.device("cuda:0"))
        y_true = sample_batched["label"].long().to(torch.device("cuda:0"))

        outputs = nn_net(x)
        loss = criterion(outputs, y_true)
        loss_total += loss.item()
        acc_total += acc_batch(outputs, y_true)

    return loss_total/n, acc_total/n


def test_maps(work_folder, db_fold,res_start, res_end, valid_list):

    ### make out folders
    time_str = time.ctime().replace(' ','_').replace(':','_')
    all_folder = work_folder + "/TESTS"+"_"+time_str+"/"

    shutil.rmtree(all_folder, ignore_errors=True)
    os.mkdir(all_folder)

    ## LOAD NETS
    cnf_weights_file = work_folder +  "/cnf_net.pth"
    clf_nn = eq_net_T4.Net('CLF NET')
    seg_nn = seg_net.Net('Seg Net', clf_net = clf_nn)
    cnf_nn = cnf_net.Net('CNF Net', seg_nn = seg_nn)

    cnf_nn.load_state_dict(torch.load(cnf_weights_file,map_location=torch.device('cpu')))
    cnf_nn.eval()

    atoms_nn = atoms_net.Net('Atoms Net', clf_net = clf_nn)



    test_nums, valid_nums, thr_dict = work_with_lists.read_test_valid_file(valid_list)

    all_maps={}

    #create results folder
    for id in test_nums+valid_nums:
        id_str = "{0:0=4d}".format(id)
        map_fold,rs = work_with_lists.get_test_map_fold_res(db_fold,id_str)
        if map_fold == None or rs >res_end or rs < res_start:
            continue
        out_map_fold = all_folder + '/EMD-'+id_str

        os.mkdir(out_map_fold)
        shutil.copyfile(map_fold +'/prot.pdb', out_map_fold+'/prot.pdb')

        all_maps[id_str] = {"inp_fold":map_fold, "out_fold":out_map_fold, "res":rs,"thr":thr_dict[id]}
        if id in test_nums:
            all_maps[id_str]["TEST/VALID"] = "TEST"
        else :
            all_maps[id_str]["TEST/VALID"] = "VALID"

    for map_id in  all_maps.keys():
        run_test_on_whole_map(cnf_nn = cnf_nn,\
        in_map_fold=all_maps[map_id]["inp_fold"], out_map_fold=all_maps[map_id]["out_fold"])
        print("FINISHED", map_id)


    return

def write_labeling_results_test_cnf(test_set = None, device_id= None,
cnf_nn = None, train_batch_num = None,background_weight = 0.05,\
max_test_batches = 500, lbl_crit = None, writer=None):
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = BATCH_SIZE_TEST, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    test_cycle = cycle(test_loader)

    res_test={}

    loss_test = 0
    acc_test  = 0
    for k in utils.label_dict.keys():
        res_test[k] = {}
        res_test[k]["tp"]=0
        res_test[k]["fp"]=0
        res_test[k]["fn"]=0

    for i_batch in range(min(len(test_loader),max_test_batches )):

        sample_batched = next(test_cycle)
        x  = sample_batched['em_map'].float()
        labels_true = sample_batched["label"].long()

        y_cnf, y_seg_pred, y_clf_pred = cnf_nn(x.cuda(device_id))
        tf_true_labels = y_true_cnf(y_seg_pred,labels_true.cuda(device_id))

        loss_per_sample = lbl_crit(y_cnf,tf_true_labels)
        sample_weight = torch.where(labels_true>0, background_weight, 1.0).cuda(device_id)
        loss =(loss_per_sample * sample_weight / sample_weight.sum()).sum()

        loss_test += loss.item()

        out_pred_np = y_cnf.to('cpu').detach().numpy()
        y_true_np =  tf_true_labels.to('cpu').detach().numpy()

        labels_np = labels_true.detach().numpy()

        true_1 = y_true_np==1
        true_0 = y_true_np==0

        out_labels_np = np.argmax(out_pred_np,1)
        acc_test += utils.acc_batch(out_pred_np, y_true_np)
        pred_1 = out_labels_np==1
        pred_0 = out_labels_np==0

        tp_all = np.logical_and(pred_1, true_1)
        fp_all = np.logical_and(pred_1, true_0)
        fn_all = np.logical_and(pred_0, true_1)

        for ky in utils.label_dict.keys():

            in_label = labels_np == utils.label_dict[ky]

            tp_lab = np.float(np.sum(np.logical_and(tp_all, in_label)))
            fp_lab = np.float(np.sum(np.logical_and(fp_all, in_label)))
            fn_lab = np.float(np.sum(np.logical_and(fn_all, in_label)))

            res_test[ky]["tp"]+=  tp_lab
            res_test[ky]["fp"]+=  fp_lab
            res_test[ky]["fn"]+=  fn_lab

    loss_test = loss_test/i_batch
    acc_test = acc_test/i_batch

    for ky in res_test.keys():
        precision = res_test[ky]["tp"]/(res_test[ky]["tp"] + res_test[ky]["fp"]+0.001)
        recall  = res_test[ky]["tp"]/(res_test[ky]["tp"] + res_test[ky]["fn"]+0.001)
        writer.add_scalars(ky, {"Prec":precision, "Rec": recall}, train_batch_num)

    writer.add_scalars("Loss", {'test':loss_test}, train_batch_num)
    writer.add_scalars("Acc",{'test':acc_test}, train_batch_num)
    writer.flush()

    return acc_test, loss_test

def calc_test_lost_acc_cnf(nn_net,criterion,test_loader):
    n = 0
    loss_total = 0
    acc_total  = 0
    for i_batch, sample_batched in enumerate(test_loader):
        n+=1
        x  = sample_batched['em_map'].float().to(torch.device("cuda:0"))
        y_true = sample_batched["label"].long().to(torch.device("cuda:0"))

        outputs = nn_net(x)
        loss = criterion(outputs, y_true)
        loss_total += loss.item()
        acc_total += acc_batch(outputs, y_true)

    return loss_total/n, acc_total/n

def y_true_cnf(seg_out, true_label):
    lab_out = torch.argmax(seg_out,1)
    tf_true = torch.eq(lab_out, true_label).long()
    tf_true.requires_grad = False
    return tf_true
