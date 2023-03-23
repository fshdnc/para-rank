#!/usr/bin/env bash


all=$(less "$1" | grep -P '^(4a?i?|3)' | wc -l)
top1=$(less "$1" | grep -P '^(4a?i?|3)\s+0' | wc -l)

python3 -c 'import sys; print(round(int(sys.argv[1])/int(sys.argv[2]), 4), sys.argv[1]+"/"+sys.argv[2], sep="\t")' $top1 $all
