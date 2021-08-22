import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys,os
import time

def get_full_list_of_patches(db_folder,thr=95):

    all_dict={}
    list_res_folders = [f for f in os.listdir(db_folder) if os.path.isdir(db_folder + '/'+f)]

    n=0

    for res_folder in list_res_folders:
        list_map_folders = [f for f in os.listdir(db_folder+'/'+res_folder ) if os.path.isdir(db_folder+'/'+res_folder + '/'+f)]
        for map_folder in list_map_folders:
            list_rot_folder = [f for f in os.listdir(db_folder+'/'+res_folder +'/' + map_folder) if os.path.isdir(db_folder+'/'+res_folder +'/' + map_folder + '/'+f)]
            for rot_folder in list_rot_folder:
                source_patch_folder = db_folder+'/'+res_folder + '/'+map_folder+'/'+rot_folder+ '/'+"patches_"+str(thr)+"/"
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

                    if n%10000==0:
                        print(str(n)+"  PATCHES FOUND")

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


def save_dict_to_list_file(target_list_file, db_folder, res_dict,thr=95):
    with open(target_list_file,"w") as f:
        for res_key in res_dict.keys():
            for map_key in res_dict[res_key].keys():
                for rot_key in res_dict[res_key][map_key].keys():
                    for p_name in res_dict[res_key][map_key][rot_key]:
                        f_name = "{0}/{1}/EMD-{2:0=4d}/rot{3}/patches_{4}/{5}".format(db_folder,res_key,map_key,rot_key,thr,p_name)
                        f.write(f_name)
                        f.write('\n')

    return

def print_list_statistics(list_file,plt_fld,plot_title):
    #os.mkdir(plt_fld)
    os.chdir(plt_fld)

    res_dict = read_db_list_file(list_file)


    res_list = np.array([x for x in res_dict.keys()])
    res_space = np.linspace(np.min(res_list)-5,np.max(res_list)+5, 10)
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

    save_dict_to_list_file(list_file_valid, DATA_FOLD, valid_res_dict,thr=95)
    save_dict_to_list_file(list_file_train, DATA_FOLD, train_res_dict,thr=95)
    save_dict_to_list_file(list_file_cand, DATA_FOLD, cand_res_dict,thr=95)

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

    save_dict_to_list_file(list_file_valid_300, DATA_FOLD, valid_res_dict_300A,thr=95)
    save_dict_to_list_file(list_file_valid_320, DATA_FOLD, valid_res_dict_320A,thr=95)
    save_dict_to_list_file(list_file_cand, DATA_FOLD, cand_res_dict,thr=95)

    print_list_statistics(list_file_valid_300,PLOTS_FOLDER,"VALID 300")
    print_list_statistics(list_file_valid_320,PLOTS_FOLDER,"VALID 320")
    print_list_statistics(list_file_cand,PLOTS_FOLDER,"cand_32A")

    return 


DB_FOLD = "//home/disk/"
DATA_FOLD = DB_FOLD +"/db/"
LIST_FOLD = DB_FOLD +"/lists/"
PLOTS_FOLDER = DB_FOLD +"/plots/"

time_suff =time.ctime().replace(" ","_").replace(":","_") 
db_fold = DB_FOLD
all_dict_in = get_full_list_of_patches(DATA_FOLD,thr=95)
print("FOLDER readed", time.ctime())
list_file = LIST_FOLD +"/all_patches.txt"
save_dict_to_list_file(list_file, DATA_FOLD, all_dict_in,thr=95)
#list_file = "/home/iscb/wolfson/Mark/data/DomainShift/db/list_files/llMon_May_17_17_01_00_2021.txt"
#all_dict_out = utils.read_db_list_file(list_file)
#print("LIST  readed",time.ctime())

#create_init_3_iter_file()

create_320A_files()
  