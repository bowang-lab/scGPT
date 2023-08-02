#!/bin/sh
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=8
#SBATCH --array=1,2,4,5,7,8
#SBATCH --mem=48G
#SBATCH --qos=nopreemption
#SBATCH -p cpu

QUERY_PATH="query_list.txt"

query_name=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $QUERY_PATH)

echo "processing ${query_name}"

DATASET="/scratch/ssd004/datasets/cellxgene/scb_strict/${query_name}/all_counts"
VOCAB_PATH="/scratch/ssd004/datasets/cellxgene/scFormer/scformer/tokenizer/default_census_vocab.json"
bash ~/.bashrc

NPROC=$SLURM_GPUS_ON_NODE
JOB_NAME="cellxgene_census_${QUERY_NAME}"
LOG_INTERVAL=2000
VALID_SIZE_OR_RATIO=0.03
MAX_LENGTH=1200
per_proc_batch_size=32
LAYERS=12
MODEL_SCALE=8
SAVE_DIR="/scratch/ssd004/datasets/cellxgene/profile_tmp"
# others, pancreas, lung, kidney, heart, blood
alias python_=~/.cache/pypoetry/virtualenvs/scformer-9yG_XnDJ-py3.9/bin/python
python_ -c "import torch; print(torch.version.cuda)"
python_ process_allcounts.py \
    --data-source $DATASET \
    --save-dir ${SAVE_DIR}/${JOB_NAME}-$(date +%b%d-%H-%M-%Y) \
    --vocab-path ${VOCAB_PATH} \
    --valid-size-or-ratio $VALID_SIZE_OR_RATIO \
    --max-seq-len $MAX_LENGTH \
    --batch-size $per_proc_batch_size \
    --eval-batch-size $(($per_proc_batch_size * 2)) \
    --nlayers $LAYERS \
    --nheads 8 \
    --embsize $((MODEL_SCALE * 64)) \
    --d-hid $((MODEL_SCALE * 64)) \
    --grad-accu-steps $((128 / $per_proc_batch_size)) \
    --epochs 2 \
    --lr 0.0001 \
    --warmup-ratio-or-step 10000 \
    --log-interval $LOG_INTERVAL \
    --save-interval $(($LOG_INTERVAL * 3)) \
    --trunc-by-sample \
    --no-cls \
    --no-cce \
    --fp16 |
    awk '{ print strftime("[%Y-%m-%d %H:%M:%S]"), $0; fflush(); }'
