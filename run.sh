#!/bin/bash
# set the number of nodes
#SBATCH --nodes 1
# set max wallclock time
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
# set number of GPUs
#SBATCH --job-name=job123
#SBATCH --gres=gpu:1


export TORCH_HOME=./
export CUDA_LAUNCH_BLOCKING=1

export PYTHONPATH=$(pwd)/src:$PYTHONPATH
export WANDB_MODE=offline
# Declare variables 
max_epochs=5
min_epochs=3
num_devices=1
batch_size=16
strategy='ddp'
num_nodes=1
# Testing with one epoch 
OMP_NUM_THREADS=12 torchrun --nnodes=$num_nodes --nproc_per_node=$num_devices --master-port=29501 src/train.py  trainer.max_epochs=$max_epochs trainer.min_epochs=$min_epochs  trainer.devices=$num_devices trainer.strategy=$strategy data.batch_size=$((batch_size * num_devices))