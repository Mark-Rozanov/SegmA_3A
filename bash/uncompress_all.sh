#!/bin/bash
for z in {331..370}
do
s1=$z

tar -xf /home/disk/db/$s1.tar.gz
rm  /home/disk/db/$s1.tar.gz

done
