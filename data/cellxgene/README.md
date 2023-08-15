# Build Training Cell Corpus from Cellxgene Census

- This documentation describes the procedure for building the pre-training cell corpus from the cellxgene census. 
- Please note that this script is designed to run on a cluster with the SLURM workload manager for parallelization.
- You may need to modify the scripts to run on your own system.
- Internet access is required for querying the cellxgene census dataset.
- The scripts referred to in this document are located in the `/data/cellxgene` directory.

## General Workflow for Cell Corpus Construction
- The general workflow is:
  
  0. (Optional) Configure the query list and query conditions.
  1. Build the cell index files based on query
  2. Download the dataset in `h5ad` chunks
  3. Transform the `h5ad` into `scb` (single-cell bank for high-performance IO)

## (Optional) Configure the Query List and Query Conditions
- If you wish to customize your pre-training dataset, you may modify the `data_config.py` file and `query_list.txt` file.
- In the `data_config.py` file,
  - `MAJOR_TISSUE_LIST` refers to the general organ system defined in the cellxgene census; it defines the resolution we used to store the cells.
  - `VERSION` refers to the version of the cellxgene census; we used the version `2023-05-08` for our experiments. You may change it to the latest/LTS version. Check out the [cellxgene census release plan](https://chanzuckerberg.github.io/cellxgene-census/cellxgene_census_docsite_data_release_info.html) for more information.
  - As we only use normal cells for pre-training, we filter the dataset by the `DISEASE` column in the cellxgene census.
  - For the `pan-cancer` model, we filter the dataset by the `DISEASE` column in the cellxgene census. The filtered cancer list is defined in the `cancer_list.txt` file. You may modify it according to your own needs.

## Build the Cell Index Files Based on Query

- We first query cells from the cellxgene census and filter the cells according to our needs.
  - `INDEX_PATH` is the path to the cell index file (to be generated), cell index is the SOMA id (unique index in cellxgene census) for each cell in the cellxgene census.
  - `QUERY_PATH` is the path to the query file; each line in the query file is a general organ system defined in the cellxgene census.
- Replace them in the following command and run it to generate the cell index file:

```{bash}
INDEX_PATH="path/to/index"
QUERY_PATH="path/to/query"

./build_soma_idx.sh $INDEX_PATH $QUERY_PATH
```

## Download the Dataset in Chunks
- We download the dataset in chunks; each chunk contains a maximum of 200000 cells, and the chunk size can be modified by changing the `MAX_PARTITION_SIZE` in the `download_partition.sh` file.
- Before running the script, you need to modify the `DATA_PATH`, `QUERY_PATH` and `INDEX_PATH` in the `array_download_partition.sh` file.
  - Keep the `INDEX_PATH` and `QUERY_PATH` consistent with the previous step.
  - `DATA_PATH` is the path to the directory to store the downloaded dataset. The resulting dataset will be stored in the `h5ad` format.
- Submit it to download the dataset (each compute node will need internet access):
```{bash}
sbatch array_download_partition.sh
```

## Build the `scb` Files
- We preprocess the dataset and then transform the `h5ad` into `scb` (single-cell bank for high-performance I/O).
- Before running the script, you need to modify the `DATA_PATH`, `OUTPUT_PATH`, `QUERY_PATH`, and `VOCAB_PATH` in the `array_build_scb.sh` file.
  - Keep the `DATA_PATH` and `QUERY_PATH` consistent with the previous step.
  - `OUTPUT_PATH` is the path to store the `scb` files.
  - `VOCAB_PATH` is the path to the vocabulary file, which is used to map the gene id to token id.
- Then simply submit the job to the cluster by:
```{bash}
sbatch array_build_scb.sh
```