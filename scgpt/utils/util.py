import functools
import json
import logging
import os
from pathlib import Path
import random
import subprocess
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import pandas as pd
from anndata import AnnData
import scib
from matplotlib import pyplot as plt
from matplotlib import axes
from IPython import get_ipython

from .. import logger


def gene_vocabulary():
    """
    Generate the gene name2id and id2name dictionaries.
    """
    pass


def set_seed(seed):
    """set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # if n_gpu > 0:
    #     torch.cuda.manual_seed_all(seed)


def add_file_handler(logger: logging.Logger, log_file_path: Path):
    """
    Add a file handler to the logger.
    """
    h = logging.FileHandler(log_file_path)

    # format showing time, name, function, and message
    formatter = logging.Formatter(
        "%(asctime)s-%(name)s-%(levelname)s-%(funcName)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    h.setFormatter(formatter)
    h.setLevel(logger.level)
    logger.addHandler(h)


def category_str2int(category_strs: List[str]) -> List[int]:
    set_category_strs = set(category_strs)
    name2id = {name: i for i, name in enumerate(set_category_strs)}
    return [name2id[name] for name in category_strs]


def isnotebook() -> bool:
    """check whether excuting in jupyter notebook."""
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return True  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def get_free_gpu():
    import subprocess
    import sys
    from io import StringIO
    import pandas as pd

    gpu_stats = subprocess.check_output(
        [
            "nvidia-smi",
            "--format=csv",
            "--query-gpu=memory.used,memory.free",
        ]
    ).decode("utf-8")
    gpu_df = pd.read_csv(
        StringIO(gpu_stats), names=["memory.used", "memory.free"], skiprows=1
    )
    print("GPU usage:\n{}".format(gpu_df))
    gpu_df["memory.free"] = gpu_df["memory.free"].map(lambda x: int(x.rstrip(" [MiB]")))
    idx = gpu_df["memory.free"].idxmax()
    print(
        "Find free GPU{} with {} free MiB".format(idx, gpu_df.iloc[idx]["memory.free"])
    )

    return idx


def get_git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()


def histogram(
    *data: List[np.ndarray],
    label: List[str] = ["train", "valid"],
    color: List[str] = ["blue", "red"],
    figsize: Tuple[int, int] = (9, 4),
    title: Optional[str] = None,
    show: bool = False,
    save: Optional[str] = None,
) -> axes.Axes:
    """
    Plot histogram of the data.

    Args:
        data (List[np.ndarray]): The data to plot.
        label (List[str]): The label of the data.
        color (List[str]): The color of the data.
        figsize (Tuple[int, int]): The size of the figure.
        title (Optional[str]): The title of the figure.
        show (bool): Whether to show the figure.
        save (Optional[str]): The path to save the figure.

    Returns:
        axes.Axes: The axes of the figure.
    """
    # show histogram of the clipped values
    assert len(data) == len(label), "The number of data and labels must be equal."

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=150)
    max_value = max(np.max(data) for data in data)
    ax.hist(
        [d.flatten() for d in data],
        bins=np.arange(0, max_value + 1, 1) + 0.5 if max_value < 60 else 60,
        label=label,
        density=True,
        histtype="bar",
        linewidth=2,
        rwidth=0.85,
        color=color,
    )
    ax.legend()
    ax.set_xlabel("counts")
    ax.set_ylabel("density")

    if title is not None:
        ax.set_title(title)

    if show:
        plt.show()

    if save is not None:
        fig.savefig(save, bbox_inches="tight")

    return ax


def _indicate_col_name(adata: AnnData, promt_str: str) -> Optional[str]:
    """
    Indicate the column name of the data.

    Args:
        adata (AnnData): The AnnData object.
        promt_str (str): The prompt string.

    Returns:
        Optional[str]: The column name.
    """
    while True:
        col_name = input(promt_str)
        if col_name == "":
            col_name = None
            break
        elif col_name in adata.var.columns:
            break
        elif col_name in adata.obs.columns:
            break
        else:
            print(f"The column {col_name} is not in the data. " f"Please input again.")

    return col_name


def find_required_colums(
    adata: AnnData,
    id: str,
    configs_dir: Union[str, Path],
    update: bool = False,
) -> List[Optional[str]]:
    """
    Find the required columns in AnnData, including celltype column, str_celltype
    column, the gene name column, and the experimental batch key.

    This function asks the user to input the required column names if the first
    time loading the data. The names are saved in the config file and will be
    automatically loaded next time.

    Args:
        adata (AnnData): The AnnData object.
        id (str): The id of the AnnData object, will be used as the file name for
            saving the config file.
        configs_dir (Union[str, Path]): The directory of saved config files.
        update (bool): Whether to update the config file.

    Returns:
        List[Optional[str]]: The required columns, including celltype_col, str_celltype_col,
            gene_col, and batch_col.
    """
    if isinstance(configs_dir, str):
        configs_dir = Path(configs_dir)

    if not configs_dir.exists():
        configs_dir.mkdir()

    config_file = configs_dir / f"{id}.json"

    if not config_file.exists() or update:
        print(
            "The config file does not exist, this may be the first time "
            "loading the data. \nPlease input the required column names."
        )
        print(adata)
        celltype_col = _indicate_col_name(
            adata,
            "Please input the celltype column name (skip if not applicable): ",
        )
        str_celltype_col = _indicate_col_name(
            adata, "Please input the str_celltype column name: "
        )
        gene_col = _indicate_col_name(adata, "Please input the gene column name: ")
        batch_col = _indicate_col_name(adata, "Please input the batch column name: ")

        config = {
            "celltype_col": celltype_col,
            "str_celltype_col": str_celltype_col,
            "gene_col": gene_col,
            "batch_col": batch_col,
        }

        with open(config_file, "w") as f:
            json.dump(config, f)

    else:
        with open(config_file, "r") as f:
            config = json.load(f)

    return [
        config["celltype_col"],
        config["str_celltype_col"],
        config["gene_col"],
        config["batch_col"],
    ]


def tensorlist2tensor(tensorlist, pad_value):
    max_len = max(len(t) for t in tensorlist)
    dtype = tensorlist[0].dtype
    device = tensorlist[0].device
    tensor = torch.zeros(len(tensorlist), max_len, dtype=dtype, device=device)
    tensor.fill_(pad_value)
    for i, t in enumerate(tensorlist):
        tensor[i, : len(t)] = t
    return tensor


def map_raw_id_to_vocab_id(
    raw_ids: Union[np.ndarray, torch.Tensor],
    gene_ids: np.ndarray,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Map some raw ids which are indices of the raw gene names to the indices of the

    Args:
        raw_ids: the raw ids to map
        gene_ids: the gene ids to map to
    """
    if isinstance(raw_ids, torch.Tensor):
        device = raw_ids.device
        dtype = raw_ids.dtype
        return_pt = True
        raw_ids = raw_ids.cpu().numpy()
    elif isinstance(raw_ids, np.ndarray):
        return_pt = False
        dtype = raw_ids.dtype
    else:
        raise ValueError(f"raw_ids must be either torch.Tensor or np.ndarray.")

    if raw_ids.ndim != 1:
        raise ValueError(f"raw_ids must be 1d, got {raw_ids.ndim}d.")

    if gene_ids.ndim != 1:
        raise ValueError(f"gene_ids must be 1d, got {gene_ids.ndim}d.")

    mapped_ids: np.ndarray = gene_ids[raw_ids]
    assert mapped_ids.shape == raw_ids.shape
    if return_pt:
        return torch.from_numpy(mapped_ids).type(dtype).to(device)
    return mapped_ids.astype(dtype)


# Wrapper for all scib metrics, we leave out some metrics like hvg_score, cell_cyvle,
# trajectory_conservation, because we only evaluate the latent embeddings here and
# these metrics are evaluating the reconstructed gene expressions or pseudotimes.
def eval_scib_metrics(
    adata: AnnData,
    batch_key: str = "str_batch",
    label_key: str = "celltype",
    notes: Optional[str] = None,
) -> Dict:
    results = scib.metrics.metrics(
        adata,
        adata_int=adata,
        batch_key=batch_key,
        label_key=label_key,
        embed="X_scGPT",
        isolated_labels_asw_=False,
        silhouette_=True,
        hvg_score_=False,
        graph_conn_=True,
        pcr_=True,
        isolated_labels_f1_=False,
        trajectory_=False,
        nmi_=True,  # use the clustering, bias to the best matching
        ari_=True,  # use the clustering, bias to the best matching
        cell_cycle_=False,
        kBET_=False,  # kBET return nan sometimes, need to examine
        ilisi_=False,
        clisi_=False,
    )

    if notes is not None:
        logger.info(f"{notes}")

    logger.info(f"{results}")

    result_dict = results[0].to_dict()
    logger.info(
        "Biological Conservation Metrics: \n"
        f"ASW (cell-type): {result_dict['ASW_label']:.4f}, graph cLISI: {result_dict['cLISI']:.4f}, "
        f"isolated label silhouette: {result_dict['isolated_label_silhouette']:.4f}, \n"
        "Batch Effect Removal Metrics: \n"
        f"PCR_batch: {result_dict['PCR_batch']:.4f}, ASW (batch): {result_dict['ASW_label/batch']:.4f}, "
        f"graph connectivity: {result_dict['graph_conn']:.4f}, graph iLISI: {result_dict['iLISI']:.4f}"
    )

    result_dict["avg_bio"] = np.mean(
        [
            result_dict["NMI_cluster/label"],
            result_dict["ARI_cluster/label"],
            result_dict["ASW_label"],
        ]
    )

    # remove nan value in result_dict
    result_dict = {k: v for k, v in result_dict.items() if not np.isnan(v)}

    return result_dict


# wrapper to make sure all methods are called only on the main process
def main_process_only(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if os.environ.get("LOCAL_RANK", "0") == "0":
            return func(*args, **kwargs)

    return wrapper


# class wrapper to make sure all methods are called only on the main process
class MainProcessOnly:
    def __init__(self, obj):
        self.obj = obj

    def __getattr__(self, name):
        attr = getattr(self.obj, name)

        if callable(attr):
            attr = main_process_only(attr)

        return attr
