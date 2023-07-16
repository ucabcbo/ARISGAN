#!/bin/bash -l

# Set the working directory to somewhere in your scratch space.  
#  This is a necessary step as compute nodes cannot write to $HOME.
#$ -wd /home/ucabcbo/Scratch/workspace

echo "++++++++++++++++++++++++"
echo $1
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

nvidia-smi

# Your work should be done in $TMPDIR 
# cd $TMPDIR

# Run the application and put the output into a file called date.txt
datetime=$(date +'%m%d-%H%M%S')
outputfolder="/home/ucabcbo/output/"$datetime"_"$1"/"
mkdir $outputfolder

source /home/ucabcbo/sis2/venv/bin/activate
if [ -z "$1" ]; then
    python /home/ucabcbo/sis2/train.py --exp $1 --out $outputfolder > $outputfolder"nohup.out"
else
    python /home/ucabcbo/sis2/train.py --exp $1 --restore $2 --out $outputfolder > $outputfolder"nohup.out"
fi

# Preferably, tar-up (archive) all output files onto the shared scratch area
tar -zcvf $HOME/Scratch/workspace/files_from_job_$JOB_ID.tar.gz $TMPDIR

# Make sure you have given enough time for the copy to complete!
