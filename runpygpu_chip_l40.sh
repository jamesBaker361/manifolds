#!/bin/bash

#SBATCH --partition=gpu        # Partition (job queue)

#SBATCH --requeue                 # Return job to the queue if preempted

#SBATCH --nodes=1                 # Number of nodes you require

#SBATCH --ntasks=1                # Total # of tasks across all nodes

#SBATCH --cpus-per-task=1         # Cores per task (>1 if multithread tasks)

#SBATCH --gres=gpu:1

#SBATCH --mem=64000                # Real memory (RAM) required (MB)

#SBATCH --time=3-00:00:00           # Total run time limit (D-HH:MM:SS)

#SBATCH --output=slurm_chip/generic/%j.out  # STDOUT output file

#SBATCH --error=slurm_chip/generic/%j.err   # STDERR output file (optional)


day=$(date +'%m/%d/%Y %R')
echo "gpu"  ${day} $SLURM_JOBID "node_list" $SLURM_NODELIST $@  "\n" >> jobs.txt
module purge
module load slurm/chip-gpu/23.11.4
#module load cudnn/7.0.3
module load   Autoconf/2.72-GCCcore-13.3.0 
module load  CMake/3.29.3-GCCcore-13.3.0
module load  CUDA/12.8.0  
module load FFmpeg/7.0.2-GCCcore-13.3.0
gcc --version
source myenv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
export TORCH_USE_CUDA_DSA="1"
export CUDA_LAUNCH_BLOCKING="1"
export TRANSFORMERS_CACHE="/umbc/rs/pi_donengel/users/jbaker15/trans_cache"
export HF_HOME="/umbc/rs/pi_donengel/users/jbaker15/trans_cache"
export HF_HUB_CACHE="/umbc/rs/pi_donengel/users/jbaker15/trans_cache"
export TORCH_CACHE="/umbc/rs/pi_donengel/users/jbaker15/torch_hub_cache"
export TORCH_HOME="/umbc/rs/pi_donengel/users/jbaker15/torch_home"
export WANDB_DIR="/umbc/rs/pi_donengel/users/jbaker15/wandb"
export WANDB_CACHE_DIR="/umbc/rs/pi_donengel/users/jbaker15/wandb_cache"
export HPS_ROOT="/umbc/rs/pi_donengel/users/jbaker15/hps-cache"
export IMAGE_REWARD_PATH="/umbc/rs/pi_donengel/users/jbaker15/reward-blob"
export IMAGE_REWARD_CONFIG="/umbc/rs/pi_donengel/users/jbaker15/ImageReward/med_config.json"
export BRAIN_DATA_DIR='/umbc/rs/pi_donengel/users/jbaker15/brain-diffuser/data'
export CUDA_LAUNCH_BLOCKING="1"
export SCIKIT_LEARN_DATA="/umbc/rs/pi_donengel/users/jbaker15/scikit-learn-data"
export BRAIN_DATA_DIR="/umbc/rs/pi_donengel/users/jbaker15/brain/data"
export TORCH_DISTRIBUTED_DEBUG="DETAIL"
export NCCL_DEBUG="INFO"
export NCCL_DEBUG_SUBSYS="ALL"
export TORCH_LOCAL_DIR="/umbc/rs/pi_donengel/users/jbaker15/local_torch"
export KAGGLEHUB_CACHE="/umbc/rs/pi_donengel/users/jbaker15/kaggle_cache"
export KAGGLE_CONFIG_DIR="/umbc/rs/pi_donengel/users/jbaker15/kaggle_config"
export SDL_VIDEODRIVER=dummy
export XDG_RUNTIME_DIR="/umbc/rs/pi_donengel/users/jbaker15/xdg_runtime"
echo "Running on: $(hostname)"
echo "Allocated GPUs:"
nvidia-smi
echo "version"
nvcc --version
srun --constraint=L40S python   $@
echo "Running on: $(hostname)"
echo "Allocated GPUs:"
nvidia-smi