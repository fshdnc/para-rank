#!/usr/bin/env bash

i="$1"

echo "---${i}---"
# print sbert model evaluated
sbert=$(grep scratch $i.out | grep -Po '\d+$')
echo "SBERT $sbert"
acc=$(grep TOP_1_ACC $i.out | perl -pe 's/TOP_1_ACC\s+//')
# grep for 
grep RESULTS $i.out | perl -pe 's/RESULTS\s+//;s/\s+/\t/g'; echo -e "$acc"
# move the results into the folder
mv $i.out results/${sbert}-${i}.out
mv $i.err results/${sbert}-${i}.err
