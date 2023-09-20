import cellxgene_census
import pandas as pd
import numpy as np
from data_config import VERSION
from typing import List
import os
import argparse


parser = argparse.ArgumentParser(
                    description='Download a given partition cell of the query in h5ad')

parser.add_argument("--query-name",
    type=str,
    required=True,
    help="query name to build the index",
)

parser.add_argument("--partition-idx",
    type=int,
    required=True,
    help="partition index to download",
)
parser.add_argument("--output-dir",
    type=str,
    required=True,
    help="Directory to store the output h5ad file",
)

parser.add_argument("--index-dir",
    type=str,
    required=True,
    help="Directory to find the index file",
)

parser.add_argument("--max-partition-size",
    type=int,
    required=True,
    help="The max partition size for each partition(chunk)",
)


args = parser.parse_args()

# print(args)




def define_partition(partition_idx, id_list, partition_size) -> List[str]:
    """
    This function is used to define the partition for each job

    partition_idx is the partition index, which is an integer, and 0 <= partition_idx <= len(id_list) // MAX_PARTITION_SIZE
    """
    i = partition_idx * partition_size
    return id_list[i:i + partition_size]


def load2list(query_name, soma_id_dir) -> List[int]:
    """
    This function is used to load the idx list from file
    """
    file_path = os.path.join(soma_id_dir, f"{query_name}.idx")
    with open(file_path, 'r') as fp:
        idx_list = fp.readlines()
    idx_list = [int(x.strip()) for x in idx_list]
    return idx_list

def download_partition(partition_idx, query_name, output_dir, index_dir, partition_size):
    """
    This function is used to download the partition_idx partition of the query_name
    """
    # define id partition
    id_list = load2list(query_name, index_dir)
    id_partition =  define_partition(partition_idx, id_list, partition_size)
    with cellxgene_census.open_soma(census_version=VERSION) as census:
        adata = cellxgene_census.get_anndata(census,    
                                            organism="Homo sapiens",
                                            obs_coords=id_partition,
                                            )
    # prepare the query dir if not exist
    query_dir = os.path.join(output_dir, query_name)
    if not os.path.exists(query_dir):
        os.makedirs(query_dir)
    query_adata_path = os.path.join(query_dir, f"partition_{partition_idx}.h5ad")
    adata.write_h5ad(query_adata_path)
    return query_adata_path

def del_partition(partition_idx, query_name, output_dir, index_dir, partition_size):
    query_dir = os.path.join(output_dir, query_name)
    query_adata_path = os.path.join(query_dir, f"partition_{partition_idx}.h5ad")
    os.remove(query_adata_path)


if __name__ == "__main__":

    download_partition(partition_idx=args.partition_idx,
                    query_name=args.query_name,
                    output_dir=args.output_dir,
                    index_dir=args.index_dir,
                    partition_size=args.max_partition_size
                    )