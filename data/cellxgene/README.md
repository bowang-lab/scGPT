# Build HCA from cell x gene census

- To build the cell atlas from the cell x gene census, run:

```{python}
INDEX_PATH="path/to/index"
DATA_PATH="path/to/data"
QUERY_PATH="path/to/query"

./build_soma_idx.sh $INDEX_PATH $QUERY_PATH
sbatch array_download.sh $INDEX_PATH $DATA_PATH $QUERY_PATH
```

```{bash}
cd /scratch/ssd004/datasets/cellxgene/
source env/bin/activate

INDEX_PATH="/scratch/ssd004/datasets/cellxgene/index"
DATA_PATH="/scratch/ssd004/datasets/cellxgene/anndata"
QUERY_PATH="query_list.txt"
./build_soma_idx.sh $INDEX_PATH $QUERY_PATH
sbatch array_download.sh $INDEX_PATH $DATA_PATH $QUERY_PATH
```

- build scBank object

```{bash}
sbatch array_build_scb_filtering.sh

```

## Cellxgene census dataset

- Cellxgene census is a publicly available collection of single-cell RNA-seq datasets from diverse sources. It encompasses more than 50 million cells, sourced from a variety of tissues and donors. For our model training and validation purposes, we drew from the Cellxgene census version dated `2023-05-08`. Our selection criteria for this dataset were narrowed to human cells and based on either the general tissue type or disease type as reported.

- To construct our tissue-specific foundation models, we selectively queried only healthy cells from seven different tissue types, including heart, kidney, lung, pancreas, blood, brain, and intestine. In total, this selection covers 22.8 million cells, with cell counts ranging from 210K to 13.2M across different tissues. The whole human model is constructed utilizing all the 35.1M no-disease cells sourced from human. For the training of the pan-cancer foundation model, we filtered the cells with disease types with cancer. This query resulted in a training dataset of 5.7M cells, representing 22 distinct cancer types.

## pretraining configs

- The workflow is:

  1. Build the cell index files based on query
  2. download the dataset in partitions(chunks)
  3. transform the `AnnData` into `scb`

- `query_list.txt` records the query for retrieving the cell atlas from the cell x gene census.
- `build_soma_idx.sh` builds index for all all healthy human cells collected by the census.
- `download_partition.sh` downloads the dataset in partitions(chunks) with the given index file, max partition size is 200000 cells per file by default.

  - designed run in job array mode, each job downloads one query.

- We exapnd the the cellxgene census vocab:
  1. inherit the order of original cellxgene vocab
  2. add the newly introduced genes from the cellxgene census
  3. generate the new vocab file in json format
- The procedure can be found in `dev_notebook/enrich_vocab.iypnb`
- The pan-cancer collection includes 22 types of cancers: the supported cancer type can be found in `cancer_list.txt`
