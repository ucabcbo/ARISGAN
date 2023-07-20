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
echo "================================================"
echo -e "Duration:\t\t"$duration_str
echo -e "Memory:\t\t\t"$2"G"
echo -e "Experiment:\t\t"$3
echo -e "Restore Checkpoint:\t"$4
echo "================================================"

experiment_root=$(grep -o '"experiment_root":[[:space:]]*"[^"]*"' environment.json | awk -F '"' '{print $4}')
experiment_path=$experiment_root$3"/experiment.json"
working_dir=$experiment_root$3"/nohup"
mkdir $working_dir

job_name=$(echo "$3" | tr '/' '_')

if test -f "$experiment_path"; then
    cd "/home/ucabcbo/sis2/"
    echo "qsub -N "$job_name" -l h_rt="$duration_str",mem="$2"G,gpu=1,tmpfs=100G -wd $working_dir -m es train.sh "$3" "$4
    qsub -N $job_name -l "h_rt="$duration_str",mem="$2"G,gpu=1,tmpfs=100G" -wd $working_dir -m es train.sh $3 $4
else
    echo "File does not exist: "$experiment_path
fi
