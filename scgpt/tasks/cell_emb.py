import json
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import scanpy as sc
import torch
from anndata import AnnData

from ..tokenizer import tokenize_and_pad_batch, GeneVocab
from ..preprocess import Preprocessor
from ..model import TransformerModel
from .. import logger

PathLike = Union[str, os.PathLike]


def get_batch_cell_embeddings(
    adata,
    cell_embedding_mode: str = "cls",
    model=None,
    vocab=None,
    max_length=1200,
    batch_size=64,
    model_configs=None,
    gene_ids=None,
    use_batch_labels=False,
) -> np.ndarray:
    """
    Get the cell embeddings for a batch of cells.

    Args:
        adata (AnnData): The AnnData object.
        gene_embs (np.ndarray): The gene embeddings, shape (len(vocab), d_emb).
        count_matrix (np.ndarray): The count matrix.

    Returns:
        np.ndarray: The cell embeddings.
    """
    count_matrix = (
        adata.layers["counts"]
        if isinstance(adata.layers["counts"], np.ndarray)
        else adata.layers["counts"].A
    )

    # gene vocabulary ids
    if gene_ids is None:
        gene_ids = np.array(adata.var["id_in_vocab"])
        assert np.all(gene_ids >= 0)

    if use_batch_labels:
        batch_ids = np.array(adata.obs["batch_id"].tolist())

    elif cell_embedding_mode == "cls":
        tokenized_all = tokenize_and_pad_batch(
            count_matrix,
            gene_ids,
            max_len=max_length,
            vocab=vocab,
            pad_token=model_configs["pad_token"],
            pad_value=model_configs["pad_value"],
            append_cls=True,  # append <cls> token at the beginning
            include_zero_gene=False,
        )
        all_gene_ids, all_values = tokenized_all["genes"], tokenized_all["values"]
        src_key_padding_mask = all_gene_ids.eq(vocab[model_configs["pad_token"]])
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            cell_embeddings = model.encode_batch(
                all_gene_ids,
                all_values.float(),
                src_key_padding_mask=src_key_padding_mask,
                batch_size=batch_size,
                batch_labels=None,
                time_step=0,
                return_np=True,
            )
        cell_embeddings = cell_embeddings / np.linalg.norm(
            cell_embeddings, axis=1, keepdims=True
        )
    else:
        raise ValueError(f"Unknown cell embedding mode: {cell_embedding_mode}")
    return cell_embeddings


def embed_data(
    adata_or_file: Union[AnnData, PathLike],
    model_dir: PathLike,
    cell_type_key: str = "cell_type",
    gene_col: str = "feature_name",
    max_length=1200,
    batch_size=64,
    obs_to_save: Optional[list] = None,
    device: Union[str, torch.device] = "cuda",
    return_new_adata: bool = False,
) -> AnnData:
    """
    Preprocess anndata and embed the data using the model.

    Args:
        adata_or_file (Union[AnnData, PathLike]): The AnnData object or the path to the
            AnnData object.
        model_dir (PathLike): The path to the model directory.
        cell_type_key (str): The key in adata.obs that contains the cell type labels.
            Defaults to "cell_type".
        gene_col (str): The column in adata.var that contains the gene names.
        max_length (int): The maximum length of the input sequence. Defaults to 1200.
        batch_size (int): The batch size for inference. Defaults to 64.
        obs_to_save (Optional[list]): The list of obs columns to save in the output adata.
            If None, will only keep the column of :attr:`cell_type_key`. Defaults to None.
        device (Union[str, torch.device]): The device to use. Defaults to "cuda".
        return_new_adata (bool): Whether to return a new AnnData object. If False, will
            add the cell embeddings to a new :attr:`adata.obsm` with key "X_scGPT".

    Returns:
        AnnData: The AnnData object with the cell embeddings.
    """
    if isinstance(adata_or_file, AnnData):
        adata = adata_or_file
    else:
        adata = sc.read_h5ad(adata_or_file)

    # verify cell type key and gene col
    assert cell_type_key in adata.obs
    if gene_col == "index":
        adata.var["index"] = adata.var.index
    else:
        assert gene_col in adata.var

    if device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            print("WARNING: CUDA is not available. Using CPU instead.")

    # LOAD MODEL
    model_dir = Path(model_dir)
    vocab_file = model_dir / "vocab.json"
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>", "<eoc>"]

    # vocabulary
    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)
    adata.var["id_in_vocab"] = [
        vocab[gene] if gene in vocab else -1 for gene in adata.var[gene_col]
    ]
    gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
    logger.info(
        f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
        f"in vocabulary of size {len(vocab)}."
    )
    adata = adata[:, adata.var["id_in_vocab"] >= 0]

    with open(model_config_file, "r") as f:
        model_configs = json.load(f)

    # Preprocessing
    preprocessor = Preprocessor(
        use_key="X",  # the key in adata.layers to use as raw data
        filter_gene_by_counts=False,  # step 1
        filter_cell_by_counts=False,  # step 2
        normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
        result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
        log1p=False,  # 4. whether to log1p the normalized data
        result_log1p_key="X_log1p",
        subset_hvg=False,
        # hvg_flavor="seurat_v3",
        binning=51,  # 6. whether to bin the raw data and to what number of bins
        result_binned_key="counts",  # the key in adata.layers to store the binned data
    )
    preprocessor(adata, batch_key=None)

    vocab.set_default_index(vocab["<pad>"])
    genes = adata.var[gene_col].tolist()
    gene_ids = np.array(vocab(genes), dtype=int)

    # all_counts = adata.layers["counts"]
    # num_of_non_zero_genes = [
    #     np.count_nonzero(all_counts[i]) for i in range(all_counts.shape[0])
    # ]
    # max_length = min(max_length, np.max(num_of_non_zero_genes) + 1)

    model = TransformerModel(
        ntoken=len(vocab),
        d_model=model_configs["embsize"],
        nhead=model_configs["nheads"],
        d_hid=model_configs["d_hid"],
        nlayers=model_configs["nlayers"],
        nlayers_cls=model_configs["n_layers_cls"],
        n_cls=1,
        vocab=vocab,
        dropout=model_configs["dropout"],
        pad_token=model_configs["pad_token"],
        pad_value=model_configs["pad_value"],
        do_mvc=True,
        do_dab=False,
        use_batch_labels=False,
        # num_batch_labels=num_batch_types,
        domain_spec_batchnorm=False,
        explicit_zero_prob=False,
        use_fast_transformer=True,
        fast_transformer_backend="flash",
        pre_norm=False,
    )

    try:
        model.load_state_dict(torch.load(model_file, map_location=device))
    except:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_file)
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    model.to(device)
    model.eval()

    # get cell embeddings
    cell_embeddings = get_batch_cell_embeddings(
        adata,
        cell_embedding_mode="cls",
        model=model,
        vocab=vocab,
        max_length=max_length,
        batch_size=batch_size,
        model_configs=model_configs,
        gene_ids=gene_ids,
        use_batch_labels=False,
    )

    if return_new_adata:
        obs_to_save = [cell_type_key] if obs_to_save is None else obs_to_save
        obs_df = adata.obs[obs_to_save]
        return sc.AnnData(X=cell_embeddings, obs=obs_df, dtype="float32")

    adata.obsm["X_scGPT"] = cell_embeddings
    return adata
