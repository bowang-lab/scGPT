#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=8
#SBATCH --array=1-9
#SBATCH --mem=96G
#SBATCH --qos=nopreemption
#SBATCH -p cpu



#QUERY_PATH="path/to/query.txt"
QUERY_PATH="/home/march/PycharmProjects/scGPT_LiuWuhao/data/query/query_list.txt"


#query_name=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $QUERY_PATH)
query_name=$(sed -n '3p' $QUERY_PATH)


#DATA_PATH="path/to/data/${query_name}"
DATA_PATH="/home/march/PycharmProjects/scGPT_LiuWuhao/data/scgpt_data/${query_name}"
#OUTPUT_PATH="path/to/output/${query_name}"
OUTPUT_PATH="/home/march/PycharmProjects/scGPT_LiuWuhao/data/h5ad_to_scb_data/${query_name}"
#VOCAB_PATH="path/to/vocab"
#VOCAB_PATH="/home/march/PycharmProjects/scGPT_LiuWuhao/data/vocab/default_gene_vocab.json"
VOCAB_PATH="/home/march/PycharmProjects/scGPT_LiuWuhao/scgpt/tokenizer/default_census_vocab.json"
#METAINFO_PATH="/home/march/PycharmProjects/scGPT_LiuWuhao/data/metainfo/metainfo.json"

echo "processing ${query_name}"
N=200000


mkdir -p $OUTPUT_PATH

echo "downloading to ${OUTPUT_PATH}"

python build_large_scale_data.py \
    --input-dir ${DATA_PATH} \
    --output-dir ${OUTPUT_PATH} \
    --vocab-file ${VOCAB_PATH} \
    --N ${N}
