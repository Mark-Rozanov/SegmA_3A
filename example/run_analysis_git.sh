
MAIN_FOLDER="/home/temp/SegmA_3A/"
EXAMPLE_FOLDER="${MAIN_FOLDER}/example/emd_11993/"
SEG_SCRIPT="${MAIN_FOLDER}/python/run_map_seg.py"
PLOT_SCRIPT="${MAIN_FOLDER}/python/single_map_analysis.py"
CH_SCRIPT="${MAIN_FOLDER}/chimera/create_patches_test.py"
CHIMERA_BIN="/home/ubuntu/.local/UCSF-Chimera64-1.16/bin/chimera"

INPUT_EM="${EXAMPLE_FOLDER}/emd_11993.map"  
INPUT_PDB="${EXAMPLE_FOLDER}/prot.pdb"
INPUT_MAP="${EXAMPLE_FOLDER}/input_map.npy"
TRUE_MAP="${EXAMPLE_FOLDER}/res/true_labels.npy"
DIST_MAP="${EXAMPLE_FOLDER}/dist_to_atoms.npy"
MAP_THR="0.0002"
PLOTS_FOLD="${EXAMPLE_FOLDER}/plots/"

#$CHIMERA_BIN --nogui $CH_SCRIPT $EXAMPLE_FOLDER 

NETS_FOLDER="${MAIN_FOLDER}/net_weights/"

SEG_MAP="${EXAMPLE_FOLDER}/res/seg_labels.npy"
CNF_MAP="${EXAMPLE_FOLDER}/res/cnf_labels.npy"

SEG_WEIGHTS="$NETS_FOLDER/seg_net.pth"
CNF_WEIGHTS="$NETS_FOLDER/cnf_net.pth"
SEG_ARGS="$EXAMPLE_FOLDER $CNF_WEIGHTS "


#docker run -it  --gpus 1 -v /home:/home  ad8c213c76c5 python $SEG_SCRIPT $SEG_ARGS
#python $SEG_SCRIPT $SEG_ARGS

PLOT_ARGS="$INPUT_MAP $TRUE_MAP $SEG_MAP $CNF_MAP $MAP_THR $DIST_MAP  $PLOTS_FOLD"

docker run -it  --gpus 1 -v /home:/home  ad8c213c76c5 python $PLOT_SCRIPT $PLOT_ARGS
#python $PLOT_SCRIPT $PLOT_ARGS
