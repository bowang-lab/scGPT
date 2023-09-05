#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --job-name=scGPT_embed_atlas
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --partition=a100
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=10-00:00:00
#SBATCH --mail-user=subercui@gmail.com
#SBATCH --mail-type=ALL

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_JOB_PARTITION"=$SLURM_JOB_PARTITION
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURM_GPUS_ON_NODE"=$SLURM_GPUS_ON_NODE
echo "SLURM_SUBMIT_DIR"=$SLURM_SUBMIT_DIR
export NCCL_IB_DISABLE=1

bash ~/.bashrc

. "/pkgs/anaconda3/etc/profile.d/conda.sh"
export PATH="/pkgs/anaconda3/bin:$PATH"

nvcc --version

conda activate ~/.conda/envs/scgpt
cd ~/scGPT/tutorials

wandb agent scformer-team/scGPT_emed_atlas/l9y6uti0
