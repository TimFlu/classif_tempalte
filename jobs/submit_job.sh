#!/bin/bash
#SBATCH --job-name="t1dexi_patchtst"
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=16
#SBATCH --output=/storage/homefs/tf24s166/code/performance_metrics_estim/logs/log.txt
#SBATCH --partition=gpu-invest
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --qos=job_gpu_koch

# Your code below this line
module load Anaconda3
module load CUDA/12.2.0
module load Workspace_Home
eval "$(conda shell.bash hook)"
/storage/homefs/tf24s166/.conda/envs/performance_metrics/bin/python /storage/homefs/tf24s166/code/cifar10/train_template.py