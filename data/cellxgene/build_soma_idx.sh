#!/bin/sh
# output directory for the index 
OUTPUT_DIR="/home/march/PycharmProjects/scGPT_LiuWuhao/data/index"
QUERY_LIST="/home/march/PycharmProjects/scGPT_LiuWuhao/data/cellxgene/query_list.txt"

while read QUERY; do
    echo "building index for ${QUERY}"
    python3 ./build_soma_idx.py --query-name ${QUERY} --output-dir ${OUTPUT_DIR} 
done < ${QUERY_LIST}

