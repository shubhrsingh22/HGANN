#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=12:00:00

# set name of job
#SBATCH --job-name=data_prep

# set number of GPUs
#SBATCH --gres=gpu:0

# run the application





#export TORCH_HOME=./
#export CUDA_LAUNCH_BLOCKING=1
#export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# Declare variables 
python hdf_prep_audioset.py