#!/bin/bash -l

# Scheduled job script for myriad

if [ -z "$1" ]; then
    echo "Error: Missing parameter. Please provide the required argument(s)."
    return 1
fi

echo "++++++++++++++++++++++++"
echo $1
echo $2
echo "++++++++++++++++++++++++"

module -f unload compilers mpi gcc-libs
module load beta-modules
module load gcc-libs/10.2.0
module load python/3.9.6-gnu-10.2.0
# module load python3/recommended
module load cuda/11.2.0/gnu-10.2.0
module load cudnn/8.1.0.77/cuda-11.2
module load tensorflow/2.11.0/gpu

module load compilers/gnu/4.9.2
module load swig/3.0.5/gnu-4.9.2
module load qt/4.8.6/gnu-4.9.2
module load ghostscript/9.19/gnu-4.9.2
module load lua/5.3.1
module load perl/5.22.0
module load graphviz/2.40.1/gnu-4.9.2

module list

source /home/ucabcbo/sis2/venv/bin/activate

nvidia-smi

timestamp=$(date +'%m%d-%H%M')

echo "Experiment: "$1
echo "Timestamp: "$timestamp
echo "Restore: "$2

experiment_root=$(grep -o '"experiment_root":[[:space:]]*"[^"]*"' /home/ucabcbo/sis2/environment.json | awk -F '"' '{print $4}')
environment=$(grep -o '"environment":[[:space:]]*"[^"]*"' /home/ucabcbo/sis2/environment.json | awk -F '"' '{print $4}')

outputfolder=$experiment_root$1"/nohup/"
outputfile=$outputfolder$timestamp"_"$environment".out"
mkdir $outputfolder

echo "Experiment Root: "$experiment_root
echo "Outputfile: "$outputfile


if [ -z "$2" ]; then
    python /home/ucabcbo/sis2/train.py --exp $1 --timestamp $timestamp > $outputfile
else
    python /home/ucabcbo/sis2/train.py --exp $1 --restore $2 --timestamp $timestamp > $outputfile
fi
