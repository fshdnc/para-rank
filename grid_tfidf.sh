#!/usr/bin/env bash

for a in char_wb char word; do
    for i in 1 2; do
	for j in 3 4 5 6; do
	    echo -n "($i, $j) $a "
	    sbatch run_tfidf.sh $i $j $a | grep -Po '\d+$'
	done
    done
done
