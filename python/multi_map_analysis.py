import numpy as np
import random
import sklearn.metrics as metrics
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics
import sys
import os
import shutil
sys.path.append("/home/iscb/wolfson/Mark/git2/python/")
from plot_utils import CNF_plots,image_CNF, calc_statistics, plot_prec_recall, plot_map_stat, plot_conf_mtrx_heat_map
from plot_utils import calc_detection_matrix,plot_aa_hbars
from work_with_lists import read_test_valid_file
from utils import label_dict

AA_colors =  {"ALA":'#8F40D4',"ARG":'#3DFF00',"ASN":'#E6E6E6',"ASP":'#BFC2C7',"CYS":'#A6A6AB',
    "GLN":'#8A99C7',"GLU":'#9C7AC7',"GLY":'#E06633',"HIS":'#F090A0',"ILE":'#50D050',"LEU":'#FF0D0D',"LYS":'#7D80B0',"MET":'#C28F8F',"PHE":'#668F8F',"PRO":'#BD80E3',
    "SER":'#A62929',"THR":'#5CB8D1',"TRP":'#00AB24',"TYR":'#C88033',"VAL":'#FFD123',"NCL":'b'}


def get_maps_res(all_patch_file, map_names):
    map_names = map_names.copy()
    res_dict = {}
    with open(all_patch_file,'r') as f:
        for ln in f.readlines():
            wrds = ln.split('/')
            for mp_name in map_names:
                if wrds[-4]==mp_name:
                    res_dict[mp_name]=int(wrds[-5])
                    map_names.remove(mp_name)
                    if len(map_names) == 0:
                        return res_dict
                    break



def calc_acc_by_label(y_lbl, true_lbl, label_names, em_map = None, em_thr = None):
    y=y_lbl.copy()
    t= true_lbl.copy()
    if em_thr!=None:
        ind_out = np.where(em_map<em_thr)
        y[ind_out] = -999
        t[ind_out] = -999

    acc_dict = {}


    for lbl_name in label_names:
        lbl  = label_names[lbl_name]
        in_true = t == lbl
        in_false = t != lbl
        in_det = y == lbl
        in_non_det = y != lbl

        TP = np.float(np.sum(np.logical_and(in_true,in_det)))
        TN = np.float(np.sum(np.logical_and(in_false,in_non_det)))
        FP = np.float(np.sum(np.logical_and(in_false,in_det)))+0.001
        FN = np.float(np.sum(np.logical_and(in_true,in_non_det)))+0.001


        acc_dict[lbl_name] = {"TP":TP,"TN":TN,"FN":FN,"FP":FP}

    return acc_dict

def get_N_samples(out_label, out_cnf, true_label, dist_matrix, em_map, map_thr, N=10000):

    in_all = np.flatnonzero(em_map>map_thr)
    in_all_sampled = random.choices(in_all, k=N)
    out_label_sampled=out_label.reshape(-1)[in_all_sampled].tolist()
    out_cnf_sampled=out_cnf.reshape(-1)[in_all_sampled].tolist()
    true_label_sampled=true_label.reshape(-1)[in_all_sampled].tolist()
    dist_mat_sampled = dist_matrix.reshape(dist_matrix.shape[0],-1)[:,in_all_sampled].transpose().tolist()

    #calc percentile

    ind_sorted = np.argsort(em_map.reshape(-1)[in_all_sampled])
    map_perc = [None]*len(ind_sorted)
    for n_perc, ind in enumerate(ind_sorted):
        map_perc[ind]=n_perc

    return out_label_sampled, out_cnf_sampled, true_label_sampled, dist_mat_sampled, map_perc




if __name__ == '__main__':

    lbls_for_plot=["ALA","ARG","ASN","ASP","GLN","GLU","GLY","HIS","ILE",\
    "LEU","LYS","MET","PHE","PRO","SER","TRP","TYR","VAL","NCL"]

    valid_file = sys.argv[1]
    map_ids = sys.argv[2]
    map_ids_cnf = sys.argv[3]
    all_patch_file = sys.argv[4]
    maps_fold = sys.argv[5]

    plots_fold = maps_fold + '/plots_by_res/'
    shutil.rmtree(plots_fold, ignore_errors=True)
    os.mkdir(plots_fold)

    test_nums, valid_nums, thr_dict = read_test_valid_file(valid_file)

    map_names  =   ['EMD-{}'.format(x) for x in map_ids.split(',')]
    map_names_cnf  =   ['EMD-{}'.format(x) for x in map_ids_cnf.split(',')]
    res_dict = get_maps_res(all_patch_file,map_names)
    res_unsorted = np.array([res_dict[mp] for mp in map_names])
    ind_sorted = np.argsort(res_unsorted)
    map_names = [map_names[x] for x in ind_sorted]
    res_sorted = res_unsorted[ind_sorted]
    map_x = [x+1 for x in range(len(map_names))]


    acc_lbl_clf = {}
    acc_lbl_seg = {}
    acc_lbl_cnf = {}
    f_scr_lbl_clf = {}
    f_scr_lbl_seg = {}
    f_scr_lbl_cnf = {}
    for l_name in lbls_for_plot:
        acc_lbl_clf[l_name] = []
        acc_lbl_seg[l_name] = []
        acc_lbl_cnf[l_name] = []

        f_scr_lbl_clf[l_name] = []
        f_scr_lbl_seg[l_name] = []
        f_scr_lbl_cnf[l_name] = []

    RES = []
    X_ax=[]

    y_lbl_seg_sampled = []
    y_lbl_cnf_sampled = []
    true_lbl_sampled = []
    dist_sampled = []
    map_percentile=[]

    for in_mp,mp in enumerate(map_names):

        print(mp)

        input_folder = maps_fold+'/'+mp
        map_thr = thr_dict[int(mp[4:])]
        map_res = res_dict[mp]
        out_pred_clf = np.load(maps_fold + "/" +mp+ "/res/clf_labels.npy")
        out_pred_seg = np.load(maps_fold + "/" +mp+ "/res/seg_labels.npy")
        out_pred_cnf = np.load(maps_fold + "/" +mp+ "/res/cnf_labels.npy")
        true_label = np.load(maps_fold + "/" +mp+ "/true_label.npy")
        em_map = np.load(maps_fold + "/" +mp+ "/input_map.npy")
        dist_mtrx = np.load(maps_fold + "/" +mp+ "/dist_to_atoms.npy")

        y_lbl_clf = np.argmax(out_pred_clf,0)
        y_lbl_seg = np.argmax(out_pred_seg,0)
        tf_label = np.argmax(out_pred_cnf,0)
        em_map_cnf = np.where(tf_label>0.5,em_map, em_map*0+map_thr-10)

        #CNF_plots(em_map, true_label, y_lbl_seg, tf_label, dist_mtrx, plots_fold)


        acc_dict_clf = calc_acc_by_label(y_lbl_clf, true_label, label_dict, em_map = em_map, em_thr = map_thr)
        acc_dict_seg = calc_acc_by_label(y_lbl_seg, true_label, label_dict, em_map = em_map, em_thr = map_thr)
        acc_dict_cnf = calc_acc_by_label(y_lbl_seg, true_label, label_dict, em_map = em_map_cnf, em_thr = map_thr)

        RES.append(res_sorted[in_mp])
        X_ax.append(map_x[in_mp])

        if mp in map_names_cnf:
            out_label_sampled_map, out_cnf_sampled_map, true_label_sampled_map, dist_sampled_map,map_percentile_maps\
             = get_N_samples(y_lbl_seg, tf_label, true_label, dist_mtrx,em_map, map_thr, N=10000)

            y_lbl_seg_sampled = y_lbl_seg_sampled + out_label_sampled_map
            y_lbl_cnf_sampled = y_lbl_cnf_sampled + out_cnf_sampled_map
            true_lbl_sampled = true_lbl_sampled + true_label_sampled_map
            dist_sampled = dist_sampled + dist_sampled_map
            map_percentile=map_percentile+map_percentile_maps

        for l_name in lbls_for_plot:
            acc_lbl_clf[l_name].append(acc_dict_clf[l_name]["TP"]/(acc_dict_clf[l_name]["TP"]+acc_dict_clf[l_name]["FP"]))
            acc_lbl_seg[l_name].append(acc_dict_seg[l_name]["TP"]/(acc_dict_seg[l_name]["TP"]+acc_dict_seg[l_name]["FP"]))
            acc_lbl_cnf[l_name].append(acc_dict_cnf[l_name]["TP"]/(acc_dict_cnf[l_name]["TP"]+acc_dict_cnf[l_name]["FP"]))

            f_scr_lbl_clf[l_name].append(acc_dict_clf[l_name]["TP"]/(acc_dict_clf[l_name]["TP"]+ 0.5*(  acc_dict_clf[l_name]["FP"]+acc_dict_clf[l_name]["FN"])))
            f_scr_lbl_seg[l_name].append(acc_dict_seg[l_name]["TP"]/(acc_dict_seg[l_name]["TP"]+0.5*( acc_dict_seg[l_name]["FP"] + acc_dict_seg[l_name]["FN"])))
            f_scr_lbl_cnf[l_name].append(acc_dict_cnf[l_name]["TP"]/(acc_dict_cnf[l_name]["TP"]+0.5*( acc_dict_cnf[l_name]["FP"] +acc_dict_cnf[l_name]["FN"])))


    y_lbl_seg_sampled = np.array(y_lbl_seg_sampled)
    true_lbl_sampled = np.array(true_lbl_sampled)
    y_lbl_cnf_sampled = np.array(y_lbl_cnf_sampled)

    in_cnf = np.flatnonzero(y_lbl_cnf_sampled>0.5)
    seg_cnf_labels = y_lbl_seg_sampled.reshape(-1)[in_cnf]
    true_cnf_labels = true_lbl_sampled.reshape(-1)[in_cnf]

    conf_matrx_seg=calc_detection_matrix(y_lbl_seg_sampled, true_lbl_sampled, 22)
    conf_matrx_cnf=calc_detection_matrix(seg_cnf_labels, true_cnf_labels, 22)
    AA_dict =  {"BKGRND":0,"ALA":1,"ARG":2,"ASN":3,"ASP":4,"CYS":5,
               "GLN":6,"GLU":7,"GLY":8,"HIS":9,"ILE":10,"LEU":11,"LYS":12,"MET":13,"PHE":14,"PRO":15,
               "SER":16,"THR":17,"TRP":18,"TYR":19,"VAL":20,"NCL":21}


    plot_conf_mtrx_heat_map(conf_matrx_seg, AA_dict, "Segmentation", plots_fold+'/CONF_SEGM.png', vmin=0, vmax=100)
    plot_conf_mtrx_heat_map(conf_matrx_cnf, AA_dict, "Confident Only", plots_fold+'/CONF_CNF.png', vmin=0, vmax=100)

    groups_def = {"BKGRND":0,"ALA":6,"ARG":2,"ASN":4,"ASP":3,"CYS":4,
               "GLN":4,"GLU":3  ,"GLY":6, "HIS":2,"ILE":6,"LEU":6,"LYS":2,"MET":4,"PHE":5,"PRO":6,
               "SER":4,"THR":4,"TRP":5,"TYR":5,"VAL":6,"NCL":1}

    group_names = {"BKGRND":0,"ALP":6,"ARM":5,"PLR":4,"NEG":3,"POS":2,"NCL":1}

    y_lbl_seg_sampled_groups = y_lbl_seg_sampled.copy()
    true_lbl_sampled_groups = true_lbl_sampled.copy()
    seg_cnf_labels_groups = seg_cnf_labels.copy()
    true_cnf_labels_groups = true_cnf_labels.copy()

    for orig_label in groups_def.keys():
        y_lbl_seg_sampled_groups = np.where(y_lbl_seg_sampled==AA_dict[orig_label],\
        groups_def[orig_label],y_lbl_seg_sampled_groups)
        true_lbl_sampled_groups = np.where(true_lbl_sampled==AA_dict[orig_label],\
        groups_def[orig_label],true_lbl_sampled_groups)
        seg_cnf_labels_groups = np.where(seg_cnf_labels==AA_dict[orig_label],\
        groups_def[orig_label],seg_cnf_labels_groups)
        true_cnf_labels_groups = np.where(true_cnf_labels==AA_dict[orig_label],\
        groups_def[orig_label],true_cnf_labels_groups)

    conf_matrx_seg_groups=calc_detection_matrix(y_lbl_seg_sampled_groups, true_lbl_sampled_groups, 7)
    conf_matrx_cnf_groups=calc_detection_matrix(seg_cnf_labels_groups, true_cnf_labels_groups, 7)
    plot_conf_mtrx_heat_map(conf_matrx_seg_groups, group_names, "Segmentation", plots_fold+'/CONF_SEGM_GRP.png', vmin=0, vmax=100)
    plot_conf_mtrx_heat_map(conf_matrx_cnf_groups, group_names, "Confident Only", plots_fold+'/CONF_CNF_GRP.png', vmin=0, vmax=100)



    CNF_plots(np.array(true_lbl_sampled), np.array(y_lbl_seg_sampled),
     np.array(y_lbl_cnf_sampled), np.array(dist_sampled), plots_fold)
    ##calculte percentage filtereds
    cnf_remove={}
    for r_type in AA_dict.keys():
        num_all = np.sum(true_lbl_sampled==AA_dict[r_type])
        num_cnf = np.sum(true_cnf_labels==AA_dict[r_type])
        N_filt = 1.0- num_cnf/(num_all+0.0001)
        cnf_remove[r_type]=N_filt*100

    plot_aa_hbars(AA_colors, cnf_remove,'Conf_Perc',plots_fold)


    for l_name in lbls_for_plot:

        plt.title(l_name + ' + Acc by resolution')
        plt.plot(X_ax, acc_lbl_clf[l_name],'bx', label='CLF-NET')
        plt.plot(X_ax, acc_lbl_seg[l_name],'rs', label='SEG-NET')
        plt.plot(X_ax, acc_lbl_cnf[l_name],'k^', label='Labeled as CONFIDENT by CNF-NET')
        for x_ind,x_val in enumerate(X_ax):
            plt.plot([x_val,x_val,x_val],[acc_lbl_clf[l_name][x_ind],acc_lbl_seg[l_name][x_ind],acc_lbl_cnf[l_name][x_ind]],'b')
            y_text = np.min([acc_lbl_clf[l_name][x_ind],acc_lbl_cnf[l_name][x_ind],acc_lbl_seg[l_name][x_ind]])-0.13
            plt.text(x_val-0.1,y_text,map_names[x_ind], rotation=90,fontsize='xx-small')

        plt.xlabel('Reported Map Resolution $\AA$' )
        plt.ylabel('Accuracy')
        plt.ylim(-0.2, 1.03)
        #plt.legend()
        plt.xticks(ticks=X_ax, labels=[str(x/100) for x in RES], fontsize ='xx-small',rotation=90 )
        plt.gca().xaxis.tick_top()
        plt.gca().tick_params(labelbottom= False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.savefig(plots_fold + '/'+l_name+'_acc.png')
        plt.close()

        plt.title(l_name + ' + F Score by resolution')
        plt.plot(X_ax, f_scr_lbl_clf[l_name],'bx', label='CLF-NET')
        plt.plot(X_ax, f_scr_lbl_seg[l_name],'rs', label='SEG-NET')
        plt.plot(X_ax, f_scr_lbl_cnf[l_name],'k^', label='Labeled as CONFIDENT by CNF-NET')
        for x_ind,x_val in enumerate(X_ax):
            plt.plot([x_val,x_val,x_val],[f_scr_lbl_clf[l_name][x_ind],f_scr_lbl_seg[l_name][x_ind],f_scr_lbl_cnf[l_name][x_ind]],'b')
            y_text = np.min([f_scr_lbl_clf[l_name][x_ind],f_scr_lbl_cnf[l_name][x_ind],f_scr_lbl_seg[l_name][x_ind]])-0.17
            plt.text(x_val-0.1,y_text,map_names[x_ind], rotation=90,fontsize='xx-small')

        plt.xlabel('Reported Map Resolution $\AA$' )
        plt.ylabel('F-SCORE')
        plt.ylim(-0.2, 1.03)
        plt.legend()
        plt.xticks(ticks=X_ax, labels=[str(x/100) for x in RES], fontsize ='xx-small',rotation=90 )
        plt.gca().xaxis.tick_top()
        plt.gca().tick_params(labelbottom= False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.savefig(plots_fold + '/'+l_name+'_fscore.png')

        plt.close()






#    RES = calc_statistics(em_mtrx,seg_labels,seg_true,conf_labels,bin_points,map_thr)
#    plot_prec_recall(RES, plots_folder)
#
#    plot_map_stat(em_mtrx=em_mtrx, labels_mtrx=seg_true, seg_mtrx=seg_labels,
#     tf_mtrx=conf_labels, plots_folder=plots_folder, thr=map_thr, bin_points=bin_points, n_bin_for_aa_plot=9)
