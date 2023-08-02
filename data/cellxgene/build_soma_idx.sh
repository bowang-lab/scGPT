#!/bin/sh
# output directory for the index 
OUTPUT_DIR=$1
QUERY_LIST=$2



while read QUERY; do
    echo "building index for ${QUERY}"
    python3 ./build_soma_idx.py --query-name ${QUERY} --output-dir ${OUTPUT_DIR} 
done < ${QUERY_LIST}

