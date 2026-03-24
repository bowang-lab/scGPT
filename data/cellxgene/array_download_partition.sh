#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=4
#SBATCH --array=1-9
#SBATCH --mem=48G
#SBATCH --qos=nopreemption
#SBATCH -p cpu



INDEX_PATH="/home/hauke.schuele/scGPT_main/data"
QUERY_PATH="/home/hauke.schuele/scGPT_main/data/cellxgene/query_list_original.txt"
DATA_PATH="/home/hauke.schuele/scGPT_main/data/cellxgene"

cd $DATA_PATH

query_name=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $QUERY_PATH)

echo "downloading ${query_name}"

./download_partition.sh ${query_name} ${INDEX_PATH} ${DATA_PATH}