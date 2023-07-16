#!/bin/sh

# $1: duration
# $2: memory
# $3: experiment
# $4: restore ckpt



IFS='.' read -ra duration <<< "$1"
hours="${duration[0]}"
decimal="${duration[1]:-0}"  # If minutes part is missing, default to 0
minutes=$((decimal * 6))
duration_str=$hours":"$minutes":0"

# Display the values
echo "Duration: "$duration_str
echo "Memory: "$2"G"
echo "Experiment: "$3
echo "Restore Checkpoint: "$4


experiment_path="/home/ucabcbo/sis2/experiments/"$3".json"

if test -f "$experiment_path"; then
    cd "/home/ucabcbo/sis2/"
    echo "qsub -N "$3" -l h_rt="$duration_str",mem="$2"G,gpu=1,tmpfs=100G -m es train.sh "$3" "$4
    # qsub -N $3 -l "h_rt="$duration_str",mem="$2"G,gpu=1,tmpfs=100G" -m es train.sh $3 $4
else
    echo "File does not exist: "$experiment_path
fi
