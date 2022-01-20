# SegmA_3A

## Example: 1 Running Segmentation and analysis of a single  map 
### Input Files
#### raw_map.mrc - The cryo-EM map
#### prot.pdb - Reference pdb file. Used only for analysis. Can be an orbitrary pdb file if the analysis is not required

### 1  Chimera Phase:
#### Load and Partition to Patches
#### Create Reference map for analysis
#### Create Distance_to_Atoms map for analysis


```bash
chimera-1.13 --nogui create_patches_test.py example/emd_11993
```

### 2 Segmentation And Confidence Phase 
```bash
python run_map_seg.py example/emd_11993   cnf_net.pth
```
### 3 Analysis Phase - Plot Graphs 
map_thr - Is the Recommended Contour value (0.00002 for EMD-11993). Can be taken from https://www.emdataresource.org/ or EMDB database

```bash
python single_map_analysis.py input_map.npy true_labels.npy seg_labels.npy map_thr cnf_labels.npy dist_to_atoms.npy example/emd_11993/plots
```
