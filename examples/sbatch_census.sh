#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --job-name=scGPT
#SBATCH --mem=150GB
#SBATCH --gres=gpu:4
#SBATCH --partition=a100
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=10-00:00:00

# log the sbatch environment
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_JOB_PARTITION"=$SLURM_JOB_PARTITION
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURM_GPUS_ON_NODE"=$SLURM_GPUS_ON_NODE
echo "SLURM_SUBMIT_DIR"=$SLURM_SUBMIT_DIR
export NCCL_IB_DISABLE=1

# . /etc/profile.d/lmod.sh
bash ~/.bashrc
nvcc --version

alias python_=path_to_python_env
python_ -c "import torch; print(torch.version.cuda)"

NPROC=$SLURM_GPUS_ON_NODE
# DATASET="path_to/datasets/3faad104-2ab8-4434-816d-474d8d2641db.scb"
# JOB_NAME="cellxgene_3faad1"
# LOG_INTERVAL=1000
# VALID_SIZE_OR_RATIO=0.1
QUERY_NAME="pan-cancer"
DATASET="/scratch/ssd004/datasets/cellxgene/scb_strict/${QUERY_NAME}/all_counts"
JOB_NAME="cellxgene_census_${QUERY_NAME}"
LOG_INTERVAL=2000
VALID_SIZE_OR_RATIO=0.03
MAX_LENGTH=1200
per_proc_batch_size=32
LAYERS=12
MODEL_SCALE=8
SAVE_DIR="/scratch/ssd004/datasets/cellxgene/save"
VOCAB_PATH="path_to/tokenizer/default_census_vocab.json"

python_ -m torch.distributed.launch \
    --nproc_per_node=$NPROC \
    pretrain.py \
    --data-source $DATASET \
    --save-dir ./save/$JOB_NAME-$(date +%b%d-%H-%M-%Y) \
    --vocab-path ${VOCAB_PATH} \
    --valid-size-or-ratio $VALID_SIZE_OR_RATIO \
    --max-seq-len $MAX_LENGTH \
    --batch-size $per_proc_batch_size \
    --eval-batch-size $(($per_proc_batch_size * 2)) \
    --nlayers $LAYERS \
    --nheads 8 \
    --embsize $((MODEL_SCALE * 64)) \
    --d-hid $((MODEL_SCALE * 64)) \
    --grad-accu-steps 1 \
    --epochs 6 \
    --lr 0.0001 \
    --warmup-ratio-or-step 10000 \
    --log-interval $LOG_INTERVAL \
    --save-interval $(($LOG_INTERVAL * 3)) \
    --trunc-by-sample \
    --no-cls \
    --no-cce \
    --fp16 |
    awk '{ print strftime("[%Y-%m-%d %H:%M:%S]"), $0; fflush(); }'
