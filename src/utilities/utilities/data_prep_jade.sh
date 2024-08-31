#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=4:00:00

# set name of job
#SBATCH --job-name=job123

# set number of GPUs
#SBATCH --gres=gpu:0

# run the application




meta_root='/jmain02/home/J2AD007/txk47/sxs27-txk47/LHGNN/jade_meta/meta'
data_root='/jmain02/flash/share/datasets/audioset'
hdf_root='/jmain02/home/J2AD007/txk47/sxs27-txk47/datasets/audioset'
datafiles_dir='/jmain02/home/J2AD007/txk47/sxs27-txk47/LHGNN/datafiles'


#export TORCH_HOME=./
#export CUDA_LAUNCH_BLOCKING=1
#export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# Declare variables 
python data_prep_audioset_jade.py "$meta_root" "$data_root" "$hdf_root" "$datafiles_dir"