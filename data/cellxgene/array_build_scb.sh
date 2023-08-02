#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=8
#SBATCH --array=8
#SBATCH --mem=96G
#SBATCH --qos=nopreemption
#SBATCH -p cpu

source ~/.cache/pypoetry/virtualenvs/scformer-9yG_XnDJ-py3.9/bin/activate
QUERY_PATH="query_list.txt"

query_name=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $QUERY_PATH)

echo "processing ${query_name}"

DATA_PATH="/scratch/ssd004/datasets/cellxgene/anndata/${query_name}"
OUTPUT_PATH="/scratch/ssd004/datasets/cellxgene/scb_strict/${query_name}"
VOCAB_PATH="/scratch/ssd004/datasets/cellxgene/scFormer/scformer/tokenizer/default_census_vocab.json"

N=200000


mkdir -p $OUTPUT_PATH

echo "downloading to ${OUTPUT_PATH}"

python build_large_scale_data.py \
    --input-dir ${DATA_PATH} \
    --output-dir ${OUTPUT_PATH} \
    --vocab-file ${VOCAB_PATH} \
    --N ${N}
