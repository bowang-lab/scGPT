#!/bin/bash
source /home/pangkuan/dev/scgpt_env/bin/activate

DATASET = "/home/pangkuan/data_disk/scb_samples/partition_0.scb"
LOG_INTERVAL = 100
per_proc_batch_size = 4

cd /home/pangkuan/dev/scGPT-release/examples
python run_pt.py \
    --data-source $DATASET \
    --save-dir ./save/eval-$(date +%b%d-%H-%M-%Y) \
    --load-model $CHECKPOINT \
    --max-seq-len $MAX_LENGTH \
    --batch-size $per_proc_batch_size \
    --eval-batch-size $(($per_proc_batch_size * 2)) \
    --epochs 0 \
    --log-interval $LOG_INTERVAL \
    --trunc-by-sample \
    --no-cls \
    --no-cce \
    --fp16