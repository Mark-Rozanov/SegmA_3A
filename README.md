# SegmA_3A

## Example Running Segmentation of a single map
### 1  Perform cropping and interpolation using Chimera

```bash
cd chimera
```
Segmentation Only - No Atomic Structure Available

```bash

chimera-1.13 --nogui preproc_ch.py ../example/emd_11993.map ../example/input_map.npy
```

### 1  Run Segmentation 

```bash
cd python
```
Segmentation on CPU 

```bash
python3.8 run_map_seg.py ../example/input_map.npy ../data/seg_net.pth ../data/cnf_net.pth ../example/out_seg.npy ../example/out_cnf.npy```
