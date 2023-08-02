#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=4
#SBATCH --array=1-9
#SBATCH --mem=48G
#SBATCH --qos=nopreemption
#SBATCH -p cpu

source /scratch/ssd004/datasets/cellxgene/env/bin/activate

cd "/scratch/ssd004/datasets/cellxgene/scFormer/census_data"

INDEX_PATH="/scratch/ssd004/datasets/cellxgene/index"
DATA_PATH="/scratch/ssd004/datasets/cellxgene/anndata"
QUERY_PATH="query_list.txt"

query_name=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $QUERY_PATH)

echo "downloading ${query_name}"

./download_partition.sh ${query_name} ${INDEX_PATH} ${DATA_PATH}