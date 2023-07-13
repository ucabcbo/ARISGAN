#!/bin/bash -l

# Request ten minutes of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=0:15:0

# Request 1 gigabyte of RAM (must be an integer followed by M, G, or T)
#$ -l mem=1G

# Request 15 gigabyte of TMPDIR space (default is 10 GB - remove if cluster is diskless)
#$ -l tmpfs=15G

# Set the name of the job.
#$ -N helloworld

# Set the working directory to somewhere in your scratch space.  
#  This is a necessary step as compute nodes cannot write to $HOME.
# Replace "<your_UCL_id>" with your UCL user ID.
#$ -wd /home/ucabcbo/Scratch/workspace

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

# Your work should be done in $TMPDIR 
cd $TMPDIR

# Run the application and put the output into a file called date.txt
source /home/ucabcbo/sis2/venv/bin/activate
python /home/ucabcbo/sis2/train.py > helloworld.txt

# Preferably, tar-up (archive) all output files onto the shared scratch area
tar -zcvf $HOME/Scratch/workspace/files_from_job_$JOB_ID.tar.gz $TMPDIR

# Make sure you have given enough time for the copy to complete!
