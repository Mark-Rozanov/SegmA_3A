import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def next_iteraion_lists(train_loss, cand_loss, new_train_file, new_cand_file , thr):

    if train_loss ==None:
        all_files = cand_loss["f_name"]
        all_losses = cand_loss["loss"].tolist()
    else:
        all_files = train_loss["f_name"]+cand_loss["f_name"]
        all_losses = train_loss["loss"].tolist()+cand_loss["loss"].tolist()

    f_cand = open(new_cand_file,"w")
    f_train = open(new_train_file,"w")

    for in_file, loss in enumerate(all_losses):
        if loss < thr:
            f_train.write(all_files[in_file])
            f_train.write("\n")
        else: 
            f_cand.write(all_files[in_file])
            f_cand.write("\n")
    f_cand.close()
    f_train.close()

    return

def read_header_line(ln):

    types = ln.split("[")[1].split("]")[0].replace("'","").split(",")
    
    acc_dict={"res":[],"res_rand":[]}

    for t in types:
        acc_dict[t]=[]

    return types, acc_dict

def get_box_plots(res, acc):
    res = np.round(np.array(res)*20)/20
    acc = np.array(acc)
    positions = np.unique(res)
    data_box = []
    for x in positions:
        data_box.append(acc[res==x])
    return data_box, positions

def load_res_file(list_file):
    loss_dict = {"res":[], "res_rand":[],"loss":[],"f_name":[]}
    with open(list_file,'r') as f:
            ln = f.readline()
            while ln != "":
                f_name, loss_str = ln.split(",")
                loss_dict["f_name"].append(f_name)

                words = f_name.split("/")
                res_name = words[-5]

                res_id = int(res_name)
                loss_dict["res"].append(res_id/100)
                seed = sum([ord(x) for x in ln])
                np.random.seed(seed)
                loss_dict["res_rand"].append((res_id + np.random.random()*4.5-2.25)/100)
                loss_data = float(loss_str)
                loss_dict["loss"].append(loss_data)
                ln = f.readline()

    loss_dict["res"] = np.array(loss_dict["res"])    
    loss_dict["loss"] = np.array(loss_dict["loss"])    
    loss_dict["res_rand"] = np.array(loss_dict["res_rand"])    
    return loss_dict

def plot_loss(ax,list_file, x_pos, crop_vals, clr):
    
    plt.axes(ax)

    loss_dict = load_res_file(list_file)
    loss_dict["loss"] = np.clip(loss_dict["loss"], crop_vals[0], crop_vals[1])
    pc = plt.violinplot(dataset = loss_dict["loss"], positions = [x_pos] ,showmeans = False,showmedians=False,showextrema=False,widths=1.5)
    pc["bodies"][0].set_facecolor(clr)
    pc["bodies"][0].set_edgecolor(clr)
    plt.boxplot(loss_dict["loss"],positions = [x_pos] ,vert=True,showfliers=False,showcaps=False,notch=True)
    N_p = len(loss_dict["loss"])
    plt.text(x_pos,crop_vals[0]-1,str(N_p),va='center',fontsize = 12)
    plt.ylim(crop_vals[0]-2,crop_vals[1]+2)
    return

def plot_resolutions(ax,list_file, x_pos,  clr):
    
    plt.axes(ax)

    loss_dict = load_res_file(list_file)
    pc = plt.violinplot(dataset = loss_dict["res_rand"], positions = [x_pos] ,showmeans = False,showmedians=False,showextrema=False,widths=1.5)
    pc["bodies"][0].set_facecolor(clr)
    pc["bodies"][0].set_edgecolor(clr)
    N_p = len(loss_dict["loss"])
    plt.text(x_pos,2,str(N_p),va='center',fontsize = 12)
    return

def analysis_300():

    positions = [1,3]

    train_list_files = ["//home/disk/res300/iter0/L_3A_train_losses.txt",
    "//home/disk/res300/iter1/L_3A_train_losses.txt"]
    valid_list_files = ["//home/disk/res300/iter0/L_3_valid_losses.txt", 
    "//home/disk/res300/iter1/L_3_valid_losses.txt"]
    cand_list_files = ["//home/disk/res300/iter0/L_3A_cand_losses.txt",
    "//home/disk/res300/iter1/L_3A_cand_losses.txt"]



    crop_values =[0, 5]


    plt.figure(figsize=(14,14))

    ax = plt.subplot(3,1,1)
    plt.ylabel('Valid')

    for inx, l in enumerate(valid_list_files):
        plot_loss(ax,l, positions[inx], crop_values, 'b')
    ax = plt.subplot(3,1,2)
    plt.ylabel('Train')
    for inx, l in enumerate(train_list_files):
        plot_loss(ax,l, positions[inx], crop_values, 'b')
    ax = plt.subplot(3,1,3)
    plt.ylabel('Candidates')
    for inx, l in enumerate(cand_list_files):
        plot_loss(ax,l, positions[inx], crop_values, 'b')

    plt.savefig("/home/disk/res300/iter1/losses.png")


    plt.figure(figsize=(14,14))

    ax = plt.subplot(3,1,1)
    plt.ylabel('Valid')
    for inx, l in enumerate(valid_list_files):
        plot_resolutions(ax,l, positions[inx], 'b')
    ax = plt.subplot(3,1,2)
    plt.ylabel('Train')
    for inx, l in enumerate(train_list_files):
        plot_resolutions(ax,l, positions[inx], 'b')
    ax = plt.subplot(3,1,3)
    plt.ylabel('Candidates')
    for inx, l in enumerate(cand_list_files):
        plot_resolutions(ax,l, positions[inx], 'b')

    plt.savefig("/home/disk/res300/iter1/resulutions.png")

    return

def analysis_320():

    crop_limits = [0, 7]

    # 320 AA
    valid_files=[]
    x_pos = []
    clrs = []
    txt = []

    valid_files.append("/home/disk/res320/iter0/L_300_valid_losses_start.txt")
    x_pos.append(1)
    clrs.append('b')
    txt.append("Iter0 - 3A")

    valid_files.append("/home/disk/res320/iter0/L_320_valid_losses_start.txt")
    x_pos.append(3)
    clrs.append('k')
    txt.append("Iter0 - 3.2A")

    valid_files.append("/home/disk/res320/iter1/L_300_valid_losses_start.txt")
    x_pos.append(6)
    clrs.append('b')
    txt.append("Iter1 - 3A")

    valid_files.append("/home/disk/res320/iter1/L_320_valid_losses_start.txt")
    x_pos.append(8)
    clrs.append('k')
    txt.append("Iter1 - 3.2A")


    valid_files.append("/home/disk/res320/iter2/L_300_valid_losses_start.txt")
    x_pos.append(11)
    clrs.append('b')
    txt.append("Iter2 - 3A")

    valid_files.append("/home/disk/res320/iter2/L_320_valid_losses_start.txt")
    x_pos.append(13)
    clrs.append('k')
    txt.append("Iter2 - 3.2A")


    plt.figure(figsize=(14,14))

    ax = plt.subplot(3,1,1)
    plt.ylabel('Valid')
    for inx, l in enumerate(valid_files):
        plot_loss(ax,l, x_pos[inx], crop_limits,clrs[inx])
    ax.set_xticks( x_pos)
    ax.set_xticklabels( txt)

    cand_files=[]
    x_pos = []
    clrs = []
    txt = []

    cand_files.append("/home/disk/res320/iter0/L_320A_cand_losses_start.txt")
    x_pos.append(1)
    clrs.append('b')
    txt.append("Cand Iter0")

    cand_files.append("/home/disk/res320/iter1/L_320_cand_losses_start.txt")
    x_pos.append(6)
    clrs.append('b')
    txt.append("Cand Iter1")

    cand_files.append("/home/disk/res320/iter1/L_320_train_losses_start.txt")
    x_pos.append(8)
    clrs.append('k')
    txt.append("Train Iter1")

    cand_files.append("/home/disk/res320/iter2/L_320_cand_losses_start.txt")
    x_pos.append(11)
    clrs.append('b')
    txt.append("Cand Iter2")

    cand_files.append("/home/disk/res320/iter2/L_320_train_losses_start.txt")
    x_pos.append(13)
    clrs.append('k')
    txt.append("Train Iter2")


    ax = plt.subplot(3,1,2)
    plt.ylabel('Candidates')
    for inx, l in enumerate(cand_files):
        plot_loss(ax,l, x_pos[inx], crop_limits,clrs[inx])
    ax.set_xticks( x_pos)
    ax.set_xticklabels( txt)

    ## plot THRESHOLDS
    valid_files=[]
    x_pos = []

    valid_files.append("/home/disk/res320/iter0/L_320_valid_losses_start.txt")
    x_pos.append([0.5,3.5])

    valid_files.append("/home/disk/res320/iter1/L_320_valid_losses_start.txt")
    x_pos.append([5.5,8.5])    
    
    valid_files.append("/home/disk/res320/iter2/L_320_valid_losses_start.txt")
    x_pos.append([10.5,13.5])

    for inx, f in enumerate(valid_files):
        loss_dict_valid=load_res_file(f)
        thr = np.quantile(loss_dict_valid["loss"],0.75)
        plt.plot(x_pos[inx],[thr,thr],'r-^')

    plt.savefig("/home/disk/res320/iter2/losses.png")

    ## NEXT ITERATION
    new_train_list_file="/home/disk/res320/iter2/L_320_train.txt"
    new_cand_list_file="/home/disk/res320/iter2/L_320_cand.txt"

    loss_dict_train=load_res_file("/home/disk/res320/iter1/L_320_train_losses_start.txt")
    loss_dict_cand=load_res_file("/home/disk/res320/iter1/L_320_cand_losses_start.txt")
    loss_dict_valid=load_res_file("/home/disk/res320/iter1/L_320_valid_losses_start.txt")
    thr = np.quantile(loss_dict_valid["loss"],0.75)
    #next_iteraion_lists(loss_dict_train, loss_dict_cand, new_train_list_file, new_cand_list_file , thr)



    return

analysis_320()
## CREATE NEW FILES
