import numpy as np
import sklearn.metrics as metrics
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics
import sys
import os
import shutil
from plot_utils import CNF_plots,image_CNF, calc_statistics, plot_prec_recall, plot_map_stat, plot_conf_mtrx_heat_map
sys.path.append("/home/temp/SegmA_3A/python/plot_utils.py")
from plot_utils import AA_dict




if __name__ == '__main__':

    bin_points = [30,40,50,60,70,80,90]+[x for x in np.arange(90.5,100,0.1)]

    input_npy_file = sys.argv[1]
    true_labels = sys.argv[2]
    seg_out_file = sys.argv[3]
    cnf_out_file = sys.argv[4]
    map_thr = np.float(sys.argv[5])
    dist_file = sys.argv[6]
    plots_folder = sys.argv[7]

    shutil.rmtree(plots_folder, ignore_errors=True)
    os.mkdir(plots_folder)


    em_mtrx     = np.load(input_npy_file)
    seg_preds   = np.load(seg_out_file)
    seg_true    = np.load(true_labels)
    conf_preds  = np.load(cnf_out_file)
    dist_mtrx   = np.load(dist_file)

    seg_labels  = np.argmax(seg_preds,0)
    conf_labels = np.argmax(conf_preds,0)


    CNF_plots(em_mtrx, seg_true, seg_labels, conf_labels, dist_mtrx, plots_folder)
    RES = calc_statistics(em_mtrx,seg_labels,seg_true,conf_labels,bin_points,map_thr)
    plot_prec_recall(RES, plots_folder)

    plot_map_stat(em_mtrx=em_mtrx, labels_mtrx=seg_true, seg_mtrx=seg_labels,
     tf_mtrx=conf_labels, plots_folder=plots_folder, thr=map_thr, bin_points=bin_points, n_bin_for_aa_plot=9)

    plot_conf_mtrx_heat_map(seg_labels, seg_true, AA_dict, "All", plots_folder+'/Conf_Matrix_All.png', vmin=0, vmax=100)
    seg_labels[conf_labels==0] = 22
    seg_true    = seg_true[em_mtrx>map_thr]
    seg_labels    = seg_labels[em_mtrx>map_thr]
    plot_conf_mtrx_heat_map(seg_labels, seg_true, AA_dict, "Above_THR_With_CONF", plots_folder+'/Above_THR_With_CONF.png', vmin=0, vmax=100)
