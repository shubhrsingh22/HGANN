#!/bin/bash
# set the number of nodes
#SBATCH --nodes 1
# set max wallclock time
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=8
# set number of GPUs
#SBATCH --job-name=audioset_train
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8

export TORCH_HOME=./
export CUDA_LAUNCH_BLOCKING=1

export PYTHONPATH=$(pwd)/src:$PYTHONPATH
export WANDB_MODE=offline
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_MIN_CHANNELS=32
# Declare variables 

max_epochs=50
min_epochs=50
num_devices=8
batch_size=32
strategy='ddp'
num_nodes=1
dataset='audioset'

if [ "$dataset" = "fsd" ]; then
    num_classes=200
     
elif [ "$dataset" = "audioset" ]; then
    num_classes=527
    
fi

# Testing with one epoch 
#OMP_NUM_THREADS=12 torchrun --nnodes=$num_nodes --nproc_per_node=$num_devices --master-port=29501 src/train.py  trainer.max_epochs=$max_epochs trainer.min_epochs=$min_epochs  trainer.devices=$num_devices trainer.strategy=$strategy data.batch_size=$((batch_size * num_devices))

# Replace the last line with:
HYDRA_FULL_ERROR=1 srun python src/train.py trainer.max_epochs=$max_epochs trainer.min_epochs=$min_epochs trainer.devices=$num_devices trainer.num_nodes=$num_nodes trainer.strategy=$strategy data.batch_size=$((batch_size * 1)) data=$dataset model.net.num_class=$num_classes 
