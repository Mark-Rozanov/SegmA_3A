#!/bin/bash
IP="54.164.68.246"
for z in {460..500}
do
s1=$z

scp -i /home/iscb/wolfson/Mark/git2/SegmA1.pem //home/iscb/wolfson/Mark/data/DomainShift/db/raw_from_web_old/raw_from_web/$s1.tar.gz ubuntu@$IP:/home/disk/db/$s1.tar.gz

done
