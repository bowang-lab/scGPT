#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=4
#SBATCH --array=1-9
#SBATCH --mem=48G
#SBATCH --qos=nopreemption
#SBATCH -p cpu



INDEX_PATH="/home/march/PycharmProjects/scGPT_LiuWuhao/data/index"
QUERY_PATH="/home/march/PycharmProjects/scGPT_LiuWuhao/data/query/query_list.txt"
DATA_PATH="/home/march/PycharmProjects/scGPT_LiuWuhao/data/scgpt_data/"

cd $DATA_PATH

#query_name=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $QUERY_PATH)
query_name=$(sed -n '3p' $QUERY_PATH)
#query_name
echo "downloading ${query_name}"

cd /home/march/PycharmProjects/scGPT_LiuWuhao/data/cellxgene/
./download_partition.sh ${query_name} ${INDEX_PATH} ${DATA_PATH}
#./download_partition.sh
