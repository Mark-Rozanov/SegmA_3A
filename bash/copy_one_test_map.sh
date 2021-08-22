#!/bin/bash
cd /home/iscb/wolfson/Mark/git2/

IP=$1

MAPNUM=$2
RES=$3

scp -i /home/iscb/wolfson/Mark/git2/SegmA1.pem //home/iscb/wolfson/Mark/data/DomainShift/db/raw_from_web_old/raw_from_web/$RES/EMD-$MAPNUM/rot0/input_map.npy ubuntu@34.238.245.222:/home/disk/db/$RES/EMD-$MAPNUM/rot0/input_map.npy
scp -i /home/iscb/wolfson/Mark/git2/SegmA1.pem //home/iscb/wolfson/Mark/data/DomainShift/db/raw_from_web_old/raw_from_web/$RES/EMD-$MAPNUM/rot1/input_map.npy ubuntu@34.238.245.222:/home/disk/db/$RES/EMD-$MAPNUM/rot1/input_map.npy
scp -i /home/iscb/wolfson/Mark/git2/SegmA1.pem //home/iscb/wolfson/Mark/data/DomainShift/db/raw_from_web_old/raw_from_web/$RES/EMD-$MAPNUM/rot2/input_map.npy ubuntu@34.238.245.222:/home/disk/db/$RES/EMD-$MAPNUM/rot2/input_map.npy
scp -i /home/iscb/wolfson/Mark/git2/SegmA1.pem //home/iscb/wolfson/Mark/data/DomainShift/db/raw_from_web_old/raw_from_web/$RES/EMD-$MAPNUM/rot3/input_map.npy ubuntu@34.238.245.222:/home/disk/db/$RES/EMD-$MAPNUM/rot3/input_map.npy
scp -i /home/iscb/wolfson/Mark/git2/SegmA1.pem //home/iscb/wolfson/Mark/data/DomainShift/db/raw_from_web_old/raw_from_web/$RES/EMD-$MAPNUM/rot4/input_map.npy ubuntu@34.238.245.222:/home/disk/db/$RES/EMD-$MAPNUM/rot4/input_map.npy
scp -i /home/iscb/wolfson/Mark/git2/SegmA1.pem //home/iscb/wolfson/Mark/data/DomainShift/db/raw_from_web_old/raw_from_web/$RES/EMD-$MAPNUM/rot5/input_map.npy ubuntu@34.238.245.222:/home/disk/db/$RES/EMD-$MAPNUM/rot5/input_map.npy
scp -i /home/iscb/wolfson/Mark/git2/SegmA1.pem //home/iscb/wolfson/Mark/data/DomainShift/db/raw_from_web_old/raw_from_web/$RES/EMD-$MAPNUM/rot6/input_map.npy ubuntu@34.238.245.222:/home/disk/db/$RES/EMD-$MAPNUM/rot6/input_map.npy
scp -i /home/iscb/wolfson/Mark/git2/SegmA1.pem //home/iscb/wolfson/Mark/data/DomainShift/db/raw_from_web_old/raw_from_web/$RES/EMD-$MAPNUM/rot7/input_map.npy ubuntu@34.238.245.222:/home/disk/db/$RES/EMD-$MAPNUM/rot7/input_map.npy
scp -i /home/iscb/wolfson/Mark/git2/SegmA1.pem //home/iscb/wolfson/Mark/data/DomainShift/db/raw_from_web_old/raw_from_web/$RES/EMD-$MAPNUM/rot8/input_map.npy ubuntu@34.238.245.222:/home/disk/db/$RES/EMD-$MAPNUM/rot8/input_map.npy
scp -i /home/iscb/wolfson/Mark/git2/SegmA1.pem //home/iscb/wolfson/Mark/data/DomainShift/db/raw_from_web_old/raw_from_web/$RES/EMD-$MAPNUM/rot9/input_map.npy ubuntu@34.238.245.222:/home/disk/db/$RES/EMD-$MAPNUM/rot9/input_map.npy

scp -i /home/iscb/wolfson/Mark/git2/SegmA1.pem //home/iscb/wolfson/Mark/data/DomainShift/db/raw_from_web_old/raw_from_web/$RES/EMD-$MAPNUM/rot0/true_label.npy ubuntu@34.238.245.222:/home/disk/db/$RES/EMD-$MAPNUM/rot0/true_label.npy
scp -i /home/iscb/wolfson/Mark/git2/SegmA1.pem //home/iscb/wolfson/Mark/data/DomainShift/db/raw_from_web_old/raw_from_web/$RES/EMD-$MAPNUM/rot1/true_label.npy ubuntu@34.238.245.222:/home/disk/db/$RES/EMD-$MAPNUM/rot1/true_label.npy
scp -i /home/iscb/wolfson/Mark/git2/SegmA1.pem //home/iscb/wolfson/Mark/data/DomainShift/db/raw_from_web_old/raw_from_web/$RES/EMD-$MAPNUM/rot2/true_label.npy ubuntu@34.238.245.222:/home/disk/db/$RES/EMD-$MAPNUM/rot2/true_label.npy
scp -i /home/iscb/wolfson/Mark/git2/SegmA1.pem //home/iscb/wolfson/Mark/data/DomainShift/db/raw_from_web_old/raw_from_web/$RES/EMD-$MAPNUM/rot3/true_label.npy ubuntu@34.238.245.222:/home/disk/db/$RES/EMD-$MAPNUM/rot3/true_label.npy
scp -i /home/iscb/wolfson/Mark/git2/SegmA1.pem //home/iscb/wolfson/Mark/data/DomainShift/db/raw_from_web_old/raw_from_web/$RES/EMD-$MAPNUM/rot4/true_label.npy ubuntu@34.238.245.222:/home/disk/db/$RES/EMD-$MAPNUM/rot4/true_label.npy
scp -i /home/iscb/wolfson/Mark/git2/SegmA1.pem //home/iscb/wolfson/Mark/data/DomainShift/db/raw_from_web_old/raw_from_web/$RES/EMD-$MAPNUM/rot5/true_label.npy ubuntu@34.238.245.222:/home/disk/db/$RES/EMD-$MAPNUM/rot5/true_label.npy
scp -i /home/iscb/wolfson/Mark/git2/SegmA1.pem //home/iscb/wolfson/Mark/data/DomainShift/db/raw_from_web_old/raw_from_web/$RES/EMD-$MAPNUM/rot6/true_label.npy ubuntu@34.238.245.222:/home/disk/db/$RES/EMD-$MAPNUM/rot6/true_label.npy
scp -i /home/iscb/wolfson/Mark/git2/SegmA1.pem //home/iscb/wolfson/Mark/data/DomainShift/db/raw_from_web_old/raw_from_web/$RES/EMD-$MAPNUM/rot7/true_label.npy ubuntu@34.238.245.222:/home/disk/db/$RES/EMD-$MAPNUM/rot7/true_label.npy
scp -i /home/iscb/wolfson/Mark/git2/SegmA1.pem //home/iscb/wolfson/Mark/data/DomainShift/db/raw_from_web_old/raw_from_web/$RES/EMD-$MAPNUM/rot8/true_label.npy ubuntu@34.238.245.222:/home/disk/db/$RES/EMD-$MAPNUM/rot8/true_label.npy
scp -i /home/iscb/wolfson/Mark/git2/SegmA1.pem //home/iscb/wolfson/Mark/data/DomainShift/db/raw_from_web_old/raw_from_web/$RES/EMD-$MAPNUM/rot9/true_label.npy ubuntu@34.238.245.222:/home/disk/db/$RES/EMD-$MAPNUM/rot9/true_label.npy

echo $MAPNUM
