import csv
import os
import sys
from pathlib import Path

import scanpy as sc

sys.path.insert(0, "../")
import scgpt as scg

model_dir = Path("../save/scGPT_human")
atlas_name = "cellxgene_cencus"

atlas_anndata_dir = Path(f"/scratch/hdd001/home/haotian/{atlas_name}_anndata")
output_dir = Path(f"/scratch/hdd001/home/haotian/{atlas_name}_embed")
file_name_csv = Path(f"./{atlas_name}_files.csv")
assert atlas_anndata_dir.exists()
output_dir.mkdir(exist_ok=True, parents=True)

# find all anndata files recursively
if file_name_csv.exists():  # read from csv
    anndata_files = []
    embed_files = []
    with open(file_name_csv, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            anndata_files.append(Path(row[0]))
            embed_files.append(Path(row[2]))
else:
    origin_tissus = [t.stem for t in list(atlas_anndata_dir.glob("*"))]
    anndata_files = sorted(list(atlas_anndata_dir.rglob("*.h5ad")))
    embed_files = []
    # write to csv
    with open(file_name_csv, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["data_file", "tissue", "embed_file"])
        for file in anndata_files:
            tissue = file.parent.stem
            if tissue in origin_tissus:
                embed_file = output_dir / tissue / f"{file.stem}.h5ad"
            else:
                tissue = ""
                embed_file = output_dir / f"{file.stem}.h5ad"
            embed_files.append(embed_file)
            writer.writerow([str(file), tissue, str(embed_file)])

# embed one file
id = 0
anndata_file = anndata_files[id]
embed_file = embed_files[id]
print(f"Embedding {anndata_file} to {embed_file}")
adata = sc.read_h5ad(anndata_file)
adata = scg.tasks.embed_data(
    adata,
    model_dir,
    cell_type_key="cell_type",
    gene_col="feature_name",
    max_length=1200,
    batch_size=128,
    obs_to_save=[
        "soma_joinid",
        "dataset_id",
        "cell_type",
        "cell_type_ontology_term_id",
        "disease",
        "tissue",
    ],
    device="cuda",
    return_new_adata=True,
)

# save
embed_file.parent.mkdir(exist_ok=True, parents=True)
adata.write_h5ad(embed_file)
