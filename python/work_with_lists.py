import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys,os
import time

time_suff =time.ctime().replace(" ","_").replace(":","_")


def get_full_list_of_patches(db_folder):

    all_dict={}
    list_res_folders = [f for f in os.listdir(db_folder) if os.path.isdir(db_folder + '/'+f)]

    n=0

    for res_folder in list_res_folders:
        list_map_folders = [f for f in os.listdir(db_folder+'/'+res_folder ) if os.path.isdir(db_folder+'/'+res_folder + '/'+f)]
        for map_folder in list_map_folders:
            list_rot_folder = [f for f in os.listdir(db_folder+'/'+res_folder +'/' + map_folder) if os.path.isdir(db_folder+'/'+res_folder +'/' + map_folder + '/'+f)]
            for rot_folder in list_rot_folder:
                source_patch_folder = db_folder+'/'+res_folder + '/'+map_folder+'/'+rot_folder+ '/'+"patches_10/"
                if not os.path.isdir(source_patch_folder):
                    continue
                f_n  = [f for f in os.listdir(source_patch_folder) if f[-10:]=='_label.npy']
                for patch_name in f_n:

                    res_id = int(res_folder)
                    map_id = int(map_folder[4:])
                    rot_id = int(rot_folder[3:])

                    res_dict = all_dict.get(res_id, {})
                    map_dict = res_dict.get(map_id, {})
                    rot_list = map_dict.get(rot_id, [])

                    rot_list.append(patch_name)

                    map_dict[rot_id] = rot_list
                    res_dict[map_id] = map_dict
                    all_dict[res_id] = res_dict
                    n+=1

                    if n % 10000==0:
                        print(str(n)+ " PATCHES READED ", time.ctime())

    return all_dict


def file_path_to_data(file_path):
    words = file_path.split("/")
    patch_name = words[-1][:-1]

    rot_name = words[-3]
    map_name = words[-4]
    res_name = words[-5]

    res_id = int(res_name)
    map_id = int(map_name[4:])
    rot_id = int(rot_name[3:])

    return patch_name,res_id, map_id, rot_id

def read_db_list_file(list_file):
    all_dict={}
    with open(list_file,'r') as f:
        ln = f.readline()
        while ln != "":
            words = ln.split("/")

            patch_name,res_id, map_id, rot_id = file_path_to_data(ln)

            res_dict = all_dict.get(res_id, {})
            map_dict = res_dict.get(map_id, {})
            rot_list = map_dict.get(rot_id, [])

            rot_list.append(patch_name)

            map_dict[rot_id] = rot_list
            res_dict[map_id] = map_dict
            all_dict[res_id] = res_dict

            ln = f.readline()

    return all_dict


def save_dict_to_list_file(target_list_file, db_folder, res_dict):
    with open(target_list_file,"w") as f:
        for res_key in res_dict.keys():
            for map_key in res_dict[res_key].keys():
                for rot_key in res_dict[res_key][map_key].keys():
                    for p_name in res_dict[res_key][map_key][rot_key]:
                        f_name = "{0}/{1}/EMD-{2:0=4d}/rot{3}/patches_10/{4}".format(db_folder,res_key,map_key,rot_key,p_name)
                        f.write(f_name)
                        f.write('\n')

    return

def print_list_statistics(list_file,plt_fld,plot_title):
    #os.mkdir(plt_fld)
    os.chdir(plt_fld)

    res_dict = read_db_list_file(list_file)


    res_list = np.array([x for x in res_dict.keys()])
    res_space = np.linspace(np.min(res_list)-5,np.max(res_list)+5, 50)
    n_files = res_space*0
    n_patches = res_space*0
    d_res = res_space[1]-res_space[0]
    for in_r,r in enumerate(res_space):
        for res_key in res_dict.keys():
            if res_key>r+d_res/2 or res_key <= r-d_res/2:
                continue
            for map_key in res_dict[res_key].keys():
                n_rot = 0
                n_patches_map = 0
                for rot_key in res_dict[res_key][map_key].keys():
                    n_rot+=1
                    for p_name in res_dict[res_key][map_key][rot_key]:
                        n_patches_map+=1
                if n_patches_map>0:
                   n_files[in_r] +=1
                   n_patches[in_r] +=n_patches_map/n_rot

    res_space = res_space/100

    plt.bar(res_space, n_files,align='center',width = d_res*0.95/100,linewidth=0.3)
    plt.xlabel('Map Resolution ' + r'$\AA$' )
    plt.ylabel('Maps in DB')
    plt.grid(True)
    plt.title(plot_title)
    plt.savefig(plot_title + "_files_vs_res" + time_suff+".png")
    plt.close()

    plt.bar(res_space, n_patches,align='center',width = d_res*0.95/100,linewidth=0.3)
    plt.xlabel('Map Resolution ' + r'$\AA$' )
    plt.ylabel('Patches in DB')
    plt.grid(True)
    plt.title(plot_title)
    plt.savefig(plot_title + "_patches" + time_suff+".png")
    plt.close()

    return

def read_test_valid_file(filename):
    test_nums=[]
    valid_nums=[]
    thr_dict = {}
    with open(filename,'r') as f:
        ln = f.readline().strip('\n') # read header
        ln = f.readline().strip('\n') #

        while ln!="TEST":
            wrds = ln.split()
            emdb_id = int(wrds[0])
            thr = float(wrds[1])
            valid_nums.append(emdb_id)
            thr_dict[emdb_id] = thr
            ln = f.readline().strip('\n')

        ln = f.readline().strip('\n')
        while ln!="":
            wrds = ln.split()
            emdb_id = int(wrds[0])
            thr = float(wrds[1])
            test_nums.append(emdb_id)
            thr_dict[emdb_id] = thr
            ln = f.readline().strip('\n')

    return test_nums, valid_nums, thr_dict


def create_init_3_iter_file():
    min_res = 2.9
    max_res = 3.1

    min_res_0 = 3.0
    max_res_0 = 3.2

    train_emdb = [6224,10845,10903,11106,11113,11889,23097,30245,30450,30525,30579,30590,30617,30639,22444]

    list_file_all = LIST_FOLD +"/all_patches.txt"

    valid_test_file = LIST_FOLD +"/test_valid.txt"


    res_dict = read_db_list_file(list_file_all)

    test_nums, valid_nums, thr_dict = read_test_valid_file(valid_test_file)

    valid_res_dict = {}
    train_res_dict = {}
    cand_res_dict = {}

    for res_key in res_dict.keys():
        if res_key>320 or res_key<300:
            continue
        for map_key in res_dict[res_key].keys():
            if map_key  in test_nums or map_key in valid_nums :
                continue
            else:
                dct = cand_res_dict.get(res_key,{})
                dct[map_key] = res_dict[res_key][map_key]
                cand_res_dict[res_key] = dct


    for res_key in res_dict.keys():
        if res_key>320 or res_key<300:
            continue
        for map_key in res_dict[res_key].keys():
            if map_key  in test_nums :
                continue
            if map_key  in valid_nums:
                dct = valid_res_dict.get(res_key,{})
                dct[map_key] = res_dict[res_key][map_key]
                valid_res_dict[res_key] = dct
            elif map_key  in train_emdb:
                dct = train_res_dict.get(res_key,{})
                dct[map_key] = res_dict[res_key][map_key]
                train_res_dict[res_key] = dct




    list_file_valid = LIST_FOLD +"/L_3_valid_"+time_suff+".txt"
    list_file_train = LIST_FOLD +"/L_3A_train_"+time_suff+".txt"
    list_file_cand  = LIST_FOLD +"/L_3A_cand_"+time_suff+".txt"

    save_dict_to_list_file(list_file_valid, DATA_FOLD, valid_res_dict)
    save_dict_to_list_file(list_file_train, DATA_FOLD, train_res_dict,)
    save_dict_to_list_file(list_file_cand, DATA_FOLD, cand_res_dict)

    print_list_statistics(list_file_all,PLOTS_FOLDER,"ALL")
    print_list_statistics(list_file_valid,PLOTS_FOLDER,"VALID ALL")
    print_list_statistics(list_file_train,PLOTS_FOLDER,"train_3A")
    print_list_statistics(list_file_cand,PLOTS_FOLDER,"cand_3A")

    return


def create_320A_files():
    min_res = 2.9
    max_res = 3.1

    min_res_0 = 3.0
    max_res_0 = 3.2


    list_file_all = LIST_FOLD +"/all_patches.txt"

    valid_test_file = LIST_FOLD +"/test_valid.txt"


    res_dict = read_db_list_file(list_file_all)
    test_nums, valid_nums, thr_dict = read_test_valid_file(valid_test_file)

    valid_res_dict_300A = {}
    valid_res_dict_320A = {}

    cand_res_dict = {}

    #CANDIDATES
    for res_key in res_dict.keys():
        if res_key>350 or res_key<320:
            continue
        for map_key in res_dict[res_key].keys():
            if map_key  in test_nums or map_key in valid_nums :
                continue
            else:
                dct = cand_res_dict.get(res_key,{})
                dct[map_key] = res_dict[res_key][map_key]
                cand_res_dict[res_key] = dct


    for res_key in res_dict.keys():
        if res_key>350 or res_key<300:
            continue
        for map_key in res_dict[res_key].keys():
            if map_key  in test_nums :
                continue
            if map_key  in valid_nums:
                if res_key>=320 and res_key<330:
                    dct = valid_res_dict_320A.get(res_key,{})
                    dct[map_key] = res_dict[res_key][map_key]
                    valid_res_dict_320A[res_key] = dct
                elif res_key>=300 and res_key<310:
                    dct = valid_res_dict_300A.get(res_key,{})
                    dct[map_key] = res_dict[res_key][map_key]
                    valid_res_dict_300A[res_key] = dct
                else:
                    continue





    list_file_valid_300 = LIST_FOLD +"/L_300_valid_"+time_suff+".txt"
    list_file_valid_320 = LIST_FOLD +"/L_320_valid_"+time_suff+".txt"
    list_file_cand  = LIST_FOLD +"/L_320A_cand_"+time_suff+".txt"

    save_dict_to_list_file(list_file_valid_300, DATA_FOLD, valid_res_dict_300A)
    save_dict_to_list_file(list_file_valid_320, DATA_FOLD, valid_res_dict_320A)
    save_dict_to_list_file(list_file_cand, DATA_FOLD, cand_res_dict)

    print_list_statistics(list_file_valid_300,PLOTS_FOLDER,"VALID 300")
    print_list_statistics(list_file_valid_320,PLOTS_FOLDER,"VALID 320")
    print_list_statistics(list_file_cand,PLOTS_FOLDER,"cand_32A")

    return

def old_functinality():

    DB_FOLD = "//home/disk/"
    DATA_FOLD = DB_FOLD +"/db/"
    LIST_FOLD = DB_FOLD +"/lists/"
    PLOTS_FOLDER = DB_FOLD +"/plots/"

    time_suff =time.ctime().replace(" ","_").replace(":","_")
    db_fold = DB_FOLD
    all_dict_in = get_full_list_of_patches(DATA_FOLD)
    print("FOLDER readed", time.ctime())
    list_file = LIST_FOLD +"/all_patches.txt"
    save_dict_to_list_file(list_file, DATA_FOLD, all_dict_in)
    #list_file = "/home/iscb/wolfson/Mark/data/DomainShift/db/list_files/llMon_May_17_17_01_00_2021.txt"
    #all_dict_out = utils.read_db_list_file(list_file)
    #print("LIST  readed",time.ctime())

    #create_init_3_iter_file()

    create_320A_files()


def create_candidates_list_file(list_all ,db_fold, res_start, res_end, valid_list, out_file_name ):
    cand_res_dict={}
    res_dict = read_db_list_file(list_all)
    test_nums, valid_nums, thr_dict = read_test_valid_file(valid_list)

    for res_key in res_dict.keys():
        if res_key>res_end or res_key<res_start:
            continue
        for map_key in res_dict[res_key].keys():
            if map_key  in test_nums or map_key in valid_nums :
                continue
            else:
                dct = cand_res_dict.get(res_key,{})
                dct[map_key] = res_dict[res_key][map_key]
                cand_res_dict[res_key] = dct
    save_dict_to_list_file(out_file_name, db_fold, cand_res_dict)



def create_valid_list_file(list_all ,db_fold, res_start, res_end, valid_list, out_file_name ):
    val_res_dict={}
    res_dict = read_db_list_file(list_all)
    test_nums, valid_nums, thr_dict = read_test_valid_file(valid_list)

    for res_key in res_dict.keys():
        if res_key>res_end or res_key<res_start:
            continue
        for map_key in res_dict[res_key].keys():
            if map_key in valid_nums :
                dct = val_res_dict.get(res_key,{})
                dct[map_key] = res_dict[res_key][map_key]
                val_res_dict[res_key] = dct

    save_dict_to_list_file(out_file_name, db_fold, val_res_dict)

def load_res_file(list_file):
    loss_dict = {"res":[], "res_rand":[],"loss":[],"f_name":[],"emd_id":[], "rot_id":[]}
    with open(list_file,'r') as f:
            ln = f.readline()
            while ln != "":
                f_name, loss_str = ln.split(",")
                loss_dict["f_name"].append(f_name)

                words = f_name.split("/")
                res_name = words[-5]
                loss_dict["emd_id"].append(words[-4].replace("EMD-",""))
                loss_dict["rot_id"].append(words[-3].replace("rot",""))

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

def next_iteraion_lists(loss_dict_train, loss_dict_cand,\
                next_train_list, next_cand_list , thr_next):

    f_cand  = open(next_cand_list,'w')
    f_train = open(next_train_list,'w')

    for in_f ,loss in enumerate(loss_dict_cand["loss"]):
        if loss>thr_next:
            f_cand.write(loss_dict_cand["f_name"][in_f])
            f_cand.write("\n")
        else:
            f_train.write(loss_dict_cand["f_name"][in_f])
            f_train.write("\n")

    if loss_dict_train == None:
        return

    for in_f ,loss in enumerate(loss_dict_train["loss"]):
        if loss>thr_next:
            f_cand.write(loss_dict_train["f_name"][in_f])
            f_cand.write("\n")
        else:
            f_train.write(loss_dict_train["f_name"][in_f])
            f_train.write("\n")
    return

def get_test_map_fold_res(db_fold,map_id_str):
    list_res_folders = [f for f in os.listdir(db_fold) if os.path.isdir(db_fold + '/'+f)]
    for res_folder in list_res_folders:
        map_fold = db_fold+'/'+res_folder + '/EMD-'+map_id_str+'/'
        map_file_orig = map_fold+'/emd_'+map_id_str +'.map'
        map_file_mrc = map_fold+'/raw_map.mrc'
        prot_file = map_fold+'/prot.pdb'

        if (os.path.isfile(map_file_orig) or os.path.isfile(map_file_mrc)) and os.path.isfile(prot_file):
            return map_fold, int(res_folder)
    return None, None
