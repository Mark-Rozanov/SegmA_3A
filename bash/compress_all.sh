#!/bin/bash
cd //home/iscb/wolfson/Mark/data/DomainShift/db/raw_from_web_old/raw_from_web/
for z in {290..500}
do
s1=$z

tar cvzf $s1.tar.gz $s1/EMD-*/rot*/patches_95

done
