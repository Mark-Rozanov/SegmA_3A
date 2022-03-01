
import sys, os
import numpy as np

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn

AA_dict =  {"BKGRND":0,"ALA":1,"ARG":2,"ASN":3,"ASP":4,"CYS":5,
"GLN":6,"GLU":7,"GLY":8,"HIS":9,"ILE":10,"LEU":11,"LYS":12,"MET":13,"PHE":14,"PRO":15,
"SER":16,"THR":17,"TRP":18,"TYR":19,"VAL":20,"NCL":21,"UNKNOWN":22}


def calc_detection_matrix(y_out, y_true, N_labels):
    det_mat = np.zeros((N_labels, N_labels),dtype=float)
    for lbl in range(N_labels):
        in_true = y_true == lbl
        n_true = np.sum(in_true)+0.001
        for lb_det in range(N_labels):
            in_det = y_out==lb_det
            det_mat[lbl,lb_det] = np.sum(in_det & in_true)/n_true
    return det_mat


def plot_conf_mtrx_heat_map(det_mat, label_names, plot_title, save_file_name, vmin=0, vmax=100):

    #det_mat = calc_detection_matrix(y_out, y_true, len(label_names.keys()))
    det_mat = np.round(det_mat*100).astype(int)

    fig, ax = plt.subplots()
    #im = ax.imshow(det_mat, vmin=vmin, vmax=vmax, interpolation = 'nearest')
    im = ax.pcolor(det_mat, cmap='turbo',vmin=vmin, vmax=vmax,edgecolors='w')

    l_names = [None]*len(label_names.keys())
    for ky in label_names.keys():
        l_names[label_names[ky]] = ky
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(l_names)))
    ax.set_yticks(np.arange(len(l_names)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(l_names)
    ax.set_yticklabels(l_names,position=(0,10))
    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)

    plt.setp(ax.get_xticklabels(),rotation=90, ha="right",rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(l_names)):
        for j in range(len(l_names)):
            text = ax.text(j+0.3, i+0.3, det_mat[i, j],ha="center", va="center", color="k",fontsize=6)

    plt.xlabel('True Label')
    plt.ylabel('Predicted Label')
    ax.set_title(plot_title)
    #fig.tight_layout()
    plt.savefig(save_file_name)
    plt.close(fig)


def normamlize_3D_box(bx, mean=0, sigma = 1):
    bx_var = np.std(bx)
    #assert bx_var>0.001
    if bx_var<0.00000001:
        bx_norm = -999*np.ones(bx.shape)
    else:
        bx_norm = (bx-np.mean(bx))/bx_var*sigma+mean
    return bx_norm



# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


#def display_point(p):
def eulerAnglesToRotationMatrix(theta) :

    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])



    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])

    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])


    R = np.dot(R_z, np.dot( R_y, R_x ))

    return R

def CNF_plots(true_labels, seg_labels, cnf_labels, dist_mtrx, fold_name):

    dist_sec = dist_mtrx.copy()

    dv=0.15
    max_values = np.amax(dist_mtrx,1,keepdims=True)
    in_max = dist_mtrx==max_values
    dist_sec[in_max]=-1

    sec_values = np.amax(dist_sec,1)
    sec_labels = np.argmax(dist_sec,1)
    max_values = np.squeeze(max_values)

    far_voxels = max_values<0.201
    close_voxels = max_values>0.5
    bound_voxels = np.logical_not(np.logical_or(far_voxels, close_voxels))

    bound_vx_bck = np.logical_and(np.logical_or(true_labels==0,sec_labels==0) , bound_voxels)
    bound_vx_atm = np.logical_and(np.logical_not(bound_vx_bck), bound_voxels)

    bound_vx_bck = np.squeeze(bound_vx_bck)
    bound_vx_atm = np.squeeze(bound_vx_atm)
    far_voxels = np.squeeze(far_voxels)
    close_voxels = np.squeeze(close_voxels)

    RES_map={}
    RES_map["Atoms"] = analyse_cnf_layer(cnf_labels, seg_labels, true_labels, close_voxels)
    RES_map["BACK"] = analyse_cnf_layer(cnf_labels, seg_labels, true_labels, far_voxels)
    RES_map["BACK_ATM"] = analyse_cnf_layer(cnf_labels, seg_labels, true_labels, bound_vx_bck)
    RES_map["ATM_ATM"] = analyse_cnf_layer(cnf_labels, seg_labels, true_labels, bound_vx_atm)

    image_CNF(RES_map["Atoms"],'within_atoms',fold_name+'within_atoms_MTRX.png')
    image_CNF(RES_map["BACK"],'back_grnd',fold_name+'back_grnd_MTRX.png')
    image_CNF(RES_map["BACK_ATM"],'bound_Atoms_Bnkgrnd',fold_name+'bound_Atoms_Bnkgrnd_MTRX.png')
    image_CNF(RES_map["ATM_ATM"],'bound_Atoms_Atoms',fold_name+'bound_Atoms_Atoms_MTRX.png')

    #REG 1

    #fpr_reg_1, tpr_reg_1, thr =   sklearn.metrics.roc_curve(y_true, y_score, pos_label=None, sample_weight=None, drop_intermediate=True)

def image_CNF(RES,ttl,fig_name):
    mtrx =np.array([[ RES["True/Conf"], RES["True/UnConf"]  ],
          [  RES["False/Conf"], RES["False/UnConf"]]]  )

    print(mtrx)

    plt.imshow(mtrx,cmap="Spectral")


    matplotlib.pyplot.text(0-0.25, 0+0.15, str(int(np.round(RES["True/Conf"]*100)))+'%',fontsize = 40)
    matplotlib.pyplot.text(0-0.25, 1+0.15, str(int(round(RES["False/Conf"]*100)))+'%',fontsize = 40)
    matplotlib.pyplot.text(1-0.25, 0+0.15, str(int(np.round(RES["True/UnConf"]*100)))+'%',fontsize = 40)
    matplotlib.pyplot.text(1-0.25, 1+0.15, str(int(round(RES["False/UnConf"]*100)))+'%',fontsize = 40)

    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    frame1.axes.yaxis.set_ticklabels([])

    matplotlib.pyplot.text(-0.50, 1.75, "CONF.",fontsize = 30)
    matplotlib.pyplot.text(0.50, 1.75, "UNCONF.",fontsize = 30)

    matplotlib.pyplot.text(-0.95, -0.25, "True",fontsize = 20)
    matplotlib.pyplot.text(-1.1, -0.05, "Labeled",fontsize = 20)
    matplotlib.pyplot.text(-0.95, 0.75, "False",fontsize = 20)
    matplotlib.pyplot.text(-1.1, 0.95, "Labeled",fontsize = 20)

    plt.title(ttl,fontsize = 20)
    plt.savefig(fig_name)
    plt.close()

def get_bins(em_mtrx,prsntg, thr):
    ind_sorted = np.argsort(em_mtrx, axis=None)
    N = len(ind_sorted)
    bns=[]
    bn_cents = []
    e = np.reshape(em_mtrx,-1)
    bns.append(e[ind_sorted[0]])
    for p in prsntg:
        n = np.int(np.round(N*p/100))
        if e[ind_sorted[n-1]]<thr:
            continue
        bns.append(e[ind_sorted[n-1]])
        bns.append(e[ind_sorted[n]])
        bn_cents.append((e[ind_sorted[n-1]]+e[ind_sorted[n]])/2)
    return bns, bn_cents

def plot_data_stat(num_ncl,num_bck, num_aa,bin_points, plots_folder):
    for k in range(len(num_ncl)) :
        s = num_ncl[k]+num_bck[k]+num_aa[k]
        num_ncl[k] = num_ncl[k]/s*100
        num_bck[k] = num_bck[k]/s*100
        num_aa[k] = num_aa[k]/s*100


    ind_for_plot = [x for x in range(len(num_ncl))]

    f = plt.figure()
    p1 = plt.bar(ind_for_plot, num_bck,color='g')
    p2 = plt.bar(ind_for_plot, num_aa,color='r',bottom=num_bck)
    a = np.array(num_bck)+np.array(num_aa)
    p3 = plt.bar(ind_for_plot, num_ncl,color='b',bottom=a)

    plt.legend((p1[-1], p2[-1],p3[-1]), ('BCKGRND', 'Amino Acid', 'Nucl. Acid'))

    plt.title('voxels by type')
    bin_points_str = ["{0:0=.3f}".format(x) for x in bin_points]
    plt.xticks(ind_for_plot, bin_points_str,rotation='vertical')

    ax = plt.gca()
    plt.savefig(plots_folder + 'data_stat.png')
    plt.close()

    return f

def plot_prec_recall(RES, plots_folder):


    for lbl in AA_dict.keys():
        res_aa_seg = RES["SEG"][lbl]
        res_aa_cnf = RES["CNF"][lbl]

        prec_seg = np.array(res_aa_seg["TP"])/(np.array(res_aa_seg["TP"])+np.array(res_aa_seg["FP"])+0.01)
        recl_seg = np.array(res_aa_seg["TP"])/(np.array(res_aa_seg["TP"])+np.array(res_aa_seg["FN"])+0.01)
        prec_cnf = np.array(res_aa_cnf["TP"])/(np.array(res_aa_cnf["TP"])+np.array(res_aa_cnf["FP"])+0.01)
        recl_cnf = np.array(res_aa_cnf["TP"])/(np.array(res_aa_cnf["TP"])+np.array(res_aa_cnf["FN"])+0.01)


        f = plt.figure()
        p1 = plt.plot(RES["bn_cents"],RES["SEG"][lbl]["Pres"],'rs-',label = "Precision: SEG NET")
        p1 = plt.plot(RES["bn_cents"],RES["SEG"][lbl]["Rcll"],'bx-',label = "Recall: SEG NET")
        p1 = plt.plot(RES["bn_cents"],RES["CNF"][lbl]["Pres"],'rd-',label = "Precision: Conf. Only")
        p1 = plt.plot(RES["bn_cents"],RES["CNF"][lbl]["Rcll"],'b*-',label = "Recall: Conf. Only")


        plt.legend()

        plt.title(lbl + 'Results')
        plt.xlabel("Map Value")

        plt.savefig(plots_folder +'/' +lbl + 'Results.png')
        plt.close()

    return f

def plot_seg_stat(num_false,num_bck_true,num_aa_true, num_ncl_true,bin_points,plots_folder):

    for k in range(len(num_false)) :
        s = num_false[k]+num_bck_true[k]+num_aa_true[k]+num_ncl_true[k]
        num_false[k] = num_false[k]/s*100
        num_aa_true[k] = num_aa_true[k]/s*100
        num_ncl_true[k] = num_ncl_true[k]/s*100
        num_bck_true[k] = num_bck_true[k]/s*100


    ind_for_plot = [x for x in range(len(num_false))]

    f = plt.figure()
    p1 = plt.bar(ind_for_plot, num_false,color='k')
    p2 = plt.bar(ind_for_plot, num_bck_true,color='g',bottom=num_false)
    a = np.array(num_false)+np.array(num_bck_true)
    p3 = plt.bar(ind_for_plot, num_aa_true,color='r',bottom=a)
    a = np.array(num_false)+np.array(num_bck_true)+np.array(num_aa_true)
    p4 = plt.bar(ind_for_plot, num_ncl_true,color='b',bottom=a)

    plt.legend((p1[-1], p2[-1],p3[-1],p4[-1]), ('False','Bckgrng True Labeled' ,'Amino Acid True Labeled', 'Nucl. Acid True Labeled'))

    plt.title('segmentation results')
    bin_points_str = ["{0:0=.3f}".format(x) for x in bin_points]
    plt.xticks(ind_for_plot, bin_points_str,rotation='vertical')
    plt.xlabel("density values")

    ax = plt.gca()
    plt.savefig(plots_folder + 'seg_stat.png')
    plt.close()
    return f

def plot_seg_stat_tf(num_false,num_bck_true,num_aa_true, num_ncl_true , num_unkn,bin_points,plots_folder):

    for k in range(len(num_false)) :
        s = num_false[k]+num_bck_true[k]+num_aa_true[k]+num_ncl_true[k]+num_unkn[k]
        num_false[k] = num_false[k]/s*100
        num_aa_true[k] = num_aa_true[k]/s*100
        num_ncl_true[k] = num_ncl_true[k]/s*100
        num_bck_true[k] = num_bck_true[k]/s*100
        num_unkn[k] = num_unkn[k]/s*100



    ind_for_plot = [x for x in range(len(num_false))]

    f = plt.figure()
    p1 = plt.bar(ind_for_plot, num_false,color='k')
    p2 = plt.bar(ind_for_plot, num_unkn,color='w',bottom=num_false, edgecolor="black", linewidth=0.5)
    a = np.array(num_false)+np.array(num_unkn)
    p3 = plt.bar(ind_for_plot, num_bck_true,color='g',bottom=a)
    a = a+np.array(num_bck_true)
    p4 = plt.bar(ind_for_plot, num_aa_true,color='r',bottom=a)
    a = a+np.array(num_aa_true)
    p5 = plt.bar(ind_for_plot, num_ncl_true,color='b',bottom=a)

    plt.legend((p1[-1], p2[-1],p3[-1],p4[-1],p5[-1]), ('False','Unknow', 'Bckgrn True Labeled' ,'Amino Acid True Labeled', 'Nucl. Acid True Labeled'))

    plt.title('segmentation results after correction')
    bin_points_str = ["{0:0=.3f}".format(x) for x in bin_points]
    plt.xticks(ind_for_plot, bin_points_str,rotation='vertical')
    plt.xlabel("density values")

    ax = plt.gca()
    #plt.setp(ax.get_xticklabels(),rotation=90, ha="right",rotation_mode="anchor")
    plt.savefig(plots_folder + 'cnf_seg_stat.png')
    plt.close()
    return f

def plot_aa_hbars(color_dict, vals_dict,ttl,plots_folder):


    fig, ax = plt.subplots()


    ks = [x for x in color_dict.keys()]
    ks.sort()
    for ink, k in enumerate(ks):
        print(ink,k,vals_dict[k],color_dict[k])
        ax.barh(ink, vals_dict[k], color = color_dict[k])
        #ax.text(-10,ink,k,fontsize =35)
    ax.set_yticks(np.arange(len(ks)))
    ax.set_yticklabels(ks,fontsize =10)
    ax.set_xticks(range(0,101,10))
    ax.set_title(ttl,fontsize =15)
    plt.tight_layout()
    plt.show()
    plt.savefig(plots_folder + 'bars_plot.png')
    plt.close()
    return fig


def calc_statistics(em_mtrx,seg_labels,seg_true,conf_labels,bin_points,thr):

    bins2,bn_cents = get_bins(em_mtrx,bin_points, thr)


    in_conf = conf_labels==1

    RES={"SEG": {},"CNF": {},"bins2":bins2,"bn_cents":bn_cents}

    for lbl in AA_dict.keys():
        RES["SEG"][lbl] = {"TP":[],"FP":[],"TN":[],"FN":[]}
        RES["CNF"][lbl] = {"TP":[],"FP":[],"TN":[],"FN":[]}

        in_label_detected = seg_labels==AA_dict[lbl]
        in_label_true = seg_true==AA_dict[lbl]

        in_TP_seg = np.logical_and(in_label_detected,in_label_true)
        in_FP_seg = np.logical_and(in_label_detected,np.logical_not(in_label_true))
        in_TN_seg = np.logical_and(np.logical_not(in_label_detected),np.logical_not(in_label_true))
        in_FN_seg = np.logical_and(np.logical_not(in_label_detected),in_label_true)

        in_TP_cnf = np.logical_and(in_conf,in_TP_seg)
        in_FP_cnf = np.logical_and(in_conf,in_FP_seg)
        in_TN_cnf = np.logical_and(in_conf,in_TN_seg)
        in_FN_cnf = np.logical_and(in_conf,in_FN_seg)

        for k in range(0,len(bins2)-1,2):
            in_range = np.logical_and(em_mtrx <bins2[k+1], em_mtrx>bins2[k])

            RES["SEG"][lbl]["TP"].append(np.sum(np.logical_and(in_TP_seg,in_range)))
            RES["SEG"][lbl]["FP"].append(np.sum(np.logical_and(in_FP_seg,in_range)))
            RES["SEG"][lbl]["TN"].append(np.sum(np.logical_and(in_TN_seg,in_range)))
            RES["SEG"][lbl]["FN"].append(np.sum(np.logical_and(in_FN_seg,in_range)))

            RES["CNF"][lbl]["TP"].append(np.sum(np.logical_and(in_TP_cnf,in_range)))
            RES["CNF"][lbl]["FP"].append(np.sum(np.logical_and(in_FP_cnf,in_range)))
            RES["CNF"][lbl]["TN"].append(np.sum(np.logical_and(in_TN_cnf,in_range)))
            RES["CNF"][lbl]["FN"].append(np.sum(np.logical_and(in_FN_cnf,in_range)))

    for lbl in AA_dict.keys():
        res_aa_seg = RES["SEG"][lbl]
        res_aa_cnf = RES["CNF"][lbl]

        RES["SEG"][lbl]["Pres"] = np.array(res_aa_seg["TP"])/(np.array(res_aa_seg["TP"])+np.array(res_aa_seg["FP"])+0.01)
        RES["SEG"][lbl]["Rcll"] = np.array(res_aa_seg["TP"])/(np.array(res_aa_seg["TP"])+np.array(res_aa_seg["FN"])+0.01)
        RES["CNF"][lbl]["Pres"] = np.array(res_aa_cnf["TP"])/(np.array(res_aa_cnf["TP"])+np.array(res_aa_cnf["FP"])+0.01)
        RES["CNF"][lbl]["Rcll"] = np.array(res_aa_cnf["TP"])/(np.array(res_aa_cnf["TP"])+np.array(res_aa_cnf["FN"])+0.01)
    return RES


def plot_map_stat(em_mtrx=None, labels_mtrx=None, seg_mtrx=None, tf_mtrx=None,\
     plots_folder=None, thr=None, bin_points=None, n_bin_for_aa_plot=None):

    bins2,bn_cents = get_bins(em_mtrx,bin_points, thr)

    AA_colors =  {"ALA":'#8F40D4',"ARG":'#3DFF00',"ASN":'#E6E6E6',"ASP":'#BFC2C7',"CYS":'#A6A6AB',
    "GLN":'#8A99C7',"GLU":'#9C7AC7',"GLY":'#E06633',"HIS":'#F090A0',"ILE":'#50D050',"LEU":'#FF0D0D',"LYS":'#7D80B0',"MET":'#C28F8F',"PHE":'#668F8F',"PRO":'#BD80E3',
    "SER":'#A62929',"THR":'#5CB8D1',"TRP":'#00AB24',"TYR":'#C88033',"VAL":'#FFD123',"NCL":'b'}

    num_ncl=[]
    num_aa = []
    num_bck = []
    num_false = []
    num_aa_true = []
    num_ncl_true = []
    num_bkgrn_true = []
    num_false_tf = []
    num_aa_true_tf = []
    num_ncl_true_tf = []
    num_bkgrn_true_tf = []
    num_unkn = []

    in_ncl = labels_mtrx == 21
    in_bck = labels_mtrx ==0
    in_aa = np.logical_and(labels_mtrx <21, labels_mtrx>0)


    in_false = seg_mtrx!=labels_mtrx
    in_true =  seg_mtrx==labels_mtrx
    in_aa_true = np.logical_and(in_aa, in_true)
    in_ncl_true = np.logical_and(in_ncl, in_true)
    in_bkgrn_true = np.logical_and(in_bck, in_true)

    in_known = tf_mtrx==1
    in_unknw = tf_mtrx==0
    in_false_tf = np.logical_and(in_false, in_known)
    in_aa_true_tf = np.logical_and(in_aa_true, in_known)
    in_ncl_true_tf = np.logical_and(in_ncl_true, in_true)
    in_bkgrn_true_tf = np.logical_and(in_bkgrn_true, in_true)


    ind_aa_pred_tf = {}
    ind_aa_true = {}

    precision_by_aa = {}

    for k in AA_dict.keys():
        ind_aa_pred_tf[k] = np.logical_and(seg_mtrx == AA_dict[k],in_known)
        ind_aa_true[k] = labels_mtrx == AA_dict[k]

    for k in range(0,len(bins2)-1,2):
        in_range = np.logical_and(em_mtrx <bins2[k+1], em_mtrx>bins2[k])
        num_ncl.append(np.sum(in_ncl[in_range]))
        num_aa.append(np.sum(in_aa[in_range]))
        num_bck.append(np.sum(in_bck[in_range]))

        num_false.append(np.sum(in_false[in_range]))
        num_bkgrn_true.append(np.sum(in_bkgrn_true[in_range]))
        num_aa_true.append(np.sum(in_aa_true[in_range]))
        num_ncl_true.append(np.sum(in_ncl_true[in_range]))

        num_false_tf.append(np.sum(in_false_tf[in_range]))
        num_aa_true_tf.append(np.sum(in_aa_true_tf[in_range]))
        num_ncl_true_tf.append(np.sum(in_ncl_true_tf[in_range]))
        num_bkgrn_true_tf.append(np.sum(in_bkgrn_true_tf[in_range]))
        num_unkn.append(np.sum(in_unknw[in_range]))

        precision_by_aa[k/2]={}
        for aa in AA_dict.keys():
            aa_bin_pred_tf = np.logical_and(in_range,ind_aa_pred_tf[aa])
            aa_bin_true = np.logical_and(in_range,ind_aa_true[aa])
            if np.sum(aa_bin_true) == 0:
                 precision_by_aa[k/2][aa]=0
            else:
                precision_by_aa[k/2][aa] = np.sum(np.logical_and(aa_bin_pred_tf,aa_bin_true))/np.sum(aa_bin_true)
            precision_by_aa[k/2][aa] = np.round(precision_by_aa[k/2][aa]*100)

    num_ncl=np.clip(np.array(num_ncl),0.001,None)
    num_aa=np.clip(np.array(num_aa),0.001,None)
    num_bck=np.clip(np.array(num_bck),0.001,None)

    num_false=np.clip(np.array(num_false),0.001,None)
    num_bkgrn_true=np.clip(np.array(num_bkgrn_true),0.001,None)
    num_aa_true=np.clip(np.array(num_aa_true),0.001,None)
    num_ncl_true=np.clip(np.array(num_ncl_true),0.001,None)

    num_false_tf=np.clip(np.array(num_false_tf),0.001,None)
    num_aa_true_tf=np.clip(np.array(num_aa_true_tf),0.001,None)
    num_ncl_true_tf=np.clip(np.array(num_ncl_true_tf),0.001,None)
    num_bkgrn_true_tf=np.clip(np.array(num_bkgrn_true_tf),0.001,None)
    num_unkn=np.clip(np.array(num_unkn),0.001,None)

    f1 = plot_data_stat(num_ncl,num_bck, num_aa,bn_cents,plots_folder)
    f2 = plot_seg_stat(num_false,num_bkgrn_true,num_aa_true,num_ncl_true,bn_cents,plots_folder)
    f3 = plot_seg_stat_tf(num_false_tf,num_bkgrn_true_tf,num_aa_true_tf,num_ncl_true_tf,num_unkn,bn_cents,plots_folder)
    f4 = plot_aa_hbars(AA_colors, precision_by_aa[n_bin_for_aa_plot],"Precision by AA type",plots_folder)

def analyse_cnf_layer(conf_labels, seg_labels, seg_true, layer_index):
    in_true = seg_labels==seg_true
    in_false = seg_labels!=seg_true
    in_conf = conf_labels==1
    in_unconf = conf_labels==0

    in_true_conf = np.logical_and(in_true,in_conf)
    in_true_unconf = np.logical_and(in_true,in_unconf)
    in_false_conf = np.logical_and(in_false,in_conf)
    in_false_unconf = np.logical_and(in_false,in_unconf)

    RES={}

    RES["True/Conf"]    = np.sum(np.logical_and(in_true_conf,layer_index))/(np.sum(layer_index)+0.001)
    RES["False/Conf"]   = np.sum(np.logical_and(in_false_conf,layer_index))/(np.sum(layer_index)+0.001)
    RES["True/UnConf"]  = np.sum(np.logical_and(in_true_unconf,layer_index))/(np.sum(layer_index)+0.001)
    RES["False/UnConf"] = np.sum(np.logical_and(in_false_unconf,layer_index))/(np.sum(layer_index)+0.001)


    return RES
