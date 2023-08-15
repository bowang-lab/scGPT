#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=8
#SBATCH --array=1-9
#SBATCH --mem=96G
#SBATCH --qos=nopreemption
#SBATCH -p cpu


QUERY_PATH="path/to/query.txt"


query_name=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $QUERY_PATH)

DATA_PATH="path/to/data/${query_name}"
OUTPUT_PATH="path/to/output/${query_name}"
VOCAB_PATH="path/to/vocab"

echo "processing ${query_name}"
N=200000


mkdir -p $OUTPUT_PATH

echo "downloading to ${OUTPUT_PATH}"

python build_large_scale_data.py \
    --input-dir ${DATA_PATH} \
    --output-dir ${OUTPUT_PATH} \
    --vocab-file ${VOCAB_PATH} \
    --N ${N}
