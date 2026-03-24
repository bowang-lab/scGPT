#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --qos=nopreemption
#SBATCH --output=/home/hauke.schuele/data_preprocessing/logs/build_scb/%x-%j.out  # File to which STDOUT will be written
#SBATCH --error=/home/hauke.schuele/data_preprocessing/logs/build_scb/%x-%j.err   # File to which STDERR will be written
#SBATCH --array=1-8
echo ""
#SBATCH -p cpu


QUERY_PATH="/home/hauke.schuele/scGPT_main/data/cellxgene/query_list_original.txt"


query_name=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $QUERY_PATH)

# DATA_PATH="/data/datasets/biology/scGPT-data/2023-05-15-disease-split"
# OUTPUT_PATH="/data/datasets/biology/scGPT-data/2023-05-15-disease-split-preprocessed-tokenizer-vocab"
# # VOCAB_PATH="/data/datasets/biology/scGPT-data/preprocessed/default_census_vocab.json"
# VOCAB_PATH="/home/hauke.schuele/scGPT_main/scgpt/tokenizer/default_census_vocab.json"
echo "processing ${query_name}"
N=200000

DATA_PATH="/data/datasets/biology/scGPT-data/2023-05-15-original/${query_name}"

OUTPUT_PATH="/data/datasets/biology/scGPT-data/2023-05-15-original-preprocessed/${query_name}"

VOCAB_PATH="/home/hauke.schuele/cellxgene_data_2023-05-15/2023-05-15-vocab.json"

module load mamba
micromamba activate scgpt_manual

mkdir -p $OUTPUT_PATH

srun python /home/hauke.schuele/scGPT_main/data/cellxgene/build_large_scale_data.py \
    --input-dir ${DATA_PATH} \
    --exclude-files ${query_name}.h5ad \
    --output-dir ${OUTPUT_PATH} \
    --vocab-file ${VOCAB_PATH} \
    --N ${N} \
