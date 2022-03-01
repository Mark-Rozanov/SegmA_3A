export ANACONDA="/home/iscb/wolfson/Mark/data/AASegTorch/anaconda3/"
export PATH="/home/iscb/wolfson/Mark/data/AASegTorch/anaconda3/bin/"

PYTHON_SCRIPT="//home/iscb/wolfson/Mark/git2/python/run_map_seg.py"
cd //home/iscb/wolfson/Mark/git2/python/

python3.8 run_map_seg.py ../example/input_map.npy ../data/seg_net.pth ../data/cnf_net.pth ../example/out_seg.npy ../example/out_cnf.npy
