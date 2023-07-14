#!/bin/sh

file_path="/home/ucabcbo/sis2/experiments/"$1".json"

if test -f "$file_path"; then
    cd "/home/ucabcbo/sis2/"
    qsub -N $1 -l "h_rt=4:00:0,mem=20G,gpu=1,tmpfs=10G" -m beas train.sh $1
else
    echo "File does not exist: "$file_path
fi
