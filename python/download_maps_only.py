import ftplib
import sys,os
import gzip
import shutil
import urllib
import urllib.request
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys,os
sys.path.append("//home/iscb/wolfson/Mark/git/work_from_home/DomainShift/python/")

import utils, utils_project
import subprocess


def search_and_get_emdb(ftp, emd_id, target_folder):
    os.chdir(target_folder)
    ftp.cwd('/pub/databases/emdb/structures/')
    data_folders = []

    ftp.dir(data_folders.append)
    for line in data_folders:
        if emd_id in line:
            fd_name = line.split()[-1]

            file_names=[]
            folder_name = '/pub/databases/emdb/structures/' +'/' +fd_name + '/map/'
            ftp.cwd(folder_name)
            ftp.dir(file_names.append)
            for line_2 in file_names:
                if emd_id in line_2:
                    file_name = line_2.split()[-1]
                    file_name_map = "raw_map.mrc" #EMD-"+emd_id + ".mrc" #file_name[:-4]
                    ftp.retrbinary("RETR " + file_name ,open(file_name, 'wb').write)
                    with gzip.open(file_name, 'rb') as f_in:
                        with open(file_name_map, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    os.remove(file_name)
    return

def download_pdb(pdbcode, datadir, downloadurl="https://files.rcsb.org/download/"):
    """
    Downloads a PDB file from the Internet and saves it in a data directory.
    :param pdbcode: The standard PDB ID e.g. '3ICB' or '3icb'
    :param datadir: The directory where the downloaded file will be saved
    :param downloadurl: The base PDB download URL, cf.
        `https://www.rcsb.org/pages/download/http#structures` for details
    :return: the full path to the downloaded PDB file or None if something went wrong
    """
    pdbfn = pdbcode + ".pdb"
    url = downloadurl + pdbfn
    outfnm = os.path.join(datadir, pdbfn)
    try:
        urllib.request.urlretrieve(url, outfnm)
        return outfnm
    except Exception as err:
        print(str(err))
        return None

INPUT_LIST_FILE="//home/iscb/wolfson/Mark/data/DomainShift/db/list_files/emdb_all_290_500.txt"
OUT_LIST_FILE="//home/iscb/wolfson/Mark/data/DomainShift/db/list_files/maps_downloaded.txt"

target_folder = "/home/iscb/wolfson/Mark/data/DomainShift/db/maps_from_emdb/"


os.chdir(target_folder)
pairs = utils_project.read_emdbdump_list_file(INPUT_LIST_FILE)
for d_row in pairs:
    try:
        print(d_row["emdID"],d_row["res"],d_row["pdbID"])
        bashCommand = "wget ftp://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-{}/map/emd_{}.map.gz".format(d_row["emdID"],d_row["emdID"])
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()


        download_pdb(d_row["pdbID"], mol_folder)
        os.rename(mol_folder + d_row["pdbID"] + ".pdb",mol_folder + "prot.pdb")
        maps_all.append(np.int(d_row["res"])/100)
    except :
        print("ERROR")
        shutil.rmtree(mol_folder)
