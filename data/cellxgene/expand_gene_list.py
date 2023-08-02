import cellxgene_census
import json

VERSION="2023-05-08"

with cellxgene_census.open_soma(census_version=VERSION) as census:
    meta_data = census["census_data"]["homo_sapiens"].ms["RNA"].var.read(
        column_names=["feature_name",],
    )
    new_gene_list = meta_data.concat().to_pandas()["feature_name"].to_list()
    # print(gene_name)

with open("/h/pangkuan/dev/scFormer/scformer/tokenizer/default_cellxgene_vocab.json", "r") as f:
    old_gene_dict = json.load(f)

print("old gene list length:", len(old_gene_dict))

expanded_dict = old_gene_dict.copy()

# count the genes in old but not in new:
# for gene, num in old_gene_dict.items():
#     if gene not in new_gene_list:
#         print(f"diff at {gene}")
        
starting_num = max(old_gene_dict.values())+1
for new_gene in new_gene_list:
    if new_gene not in old_gene_dict.keys():
        expanded_dict[new_gene] = starting_num
        starting_num += 1
print("new gene dict length:", len(expanded_dict))

dump_path = "/h/pangkuan/dev/scFormer/scformer/tokenizer/default_census_vocab.json"

with open(dump_path, "w") as f:
    json.dump(expanded_dict, f, indent=2)


