### This script is used to retrieve cell soma ids from cellxgene census

import cellxgene_census
from data_config import VALUE_FILTER, VERSION
from typing import List
import os
import argparse

parser = argparse.ArgumentParser(
                    description='Build soma index list based on query')


parser.add_argument("--query-name",
    type=str,
    required=True,
    help="query name to build the index",
)

parser.add_argument("--output-dir",
    type=str,
    required=True,
    help="Directory to store the output idx file",
)

args = parser.parse_args()
# print(args)


def retrieve_soma_idx(query_name) -> List[str]:
    """
    This function is used to retrieve cell soma ids from cellxgene census based on the query name
    """

    with cellxgene_census.open_soma(census_version=VERSION) as census:
        cell_metadata = census["census_data"]["homo_sapiens"].obs.read(
        value_filter = VALUE_FILTER[query_name],
        column_names = ["soma_joinid"]
    )
    cell_metadata = cell_metadata.concat()
    cell_metadata = cell_metadata.to_pandas()
    return cell_metadata["soma_joinid"].to_list()

def convert2file(idx_list: List[str], query_name: str, output_dir: str) -> None:
    """
    This function is used to convert the retrieved idx_list to file by query_name
    """

    # set up the dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = os.path.join(output_dir, f"{query_name}.idx")

    # write to the file
    with open(file_path, 'w') as fp:
        for item in idx_list:
            fp.write("%s\n" % item)

def build_soma_idx(query_name, output_dir) -> None:
    """
    This function is used to build the soma idx for cells under query_name
    """
    idx_list = retrieve_soma_idx(query_name)
    convert2file(idx_list, query_name, output_dir)


# if __name__ ==  "__main__":
#     build_soma_idx("heart")

build_soma_idx(args.query_name, args.output_dir)


