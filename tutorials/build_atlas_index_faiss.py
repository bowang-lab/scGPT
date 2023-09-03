# This script is used to build the index for similarity search using faiss. It supports loading embeddings and meta labels from directories and add to faiss index. The index is saved to disk for later use.
# Options can be set to custom the building process, including:
#     - embedding_dir: the directory to the embedding files
#     - meta_dir: the directory to the meta files
#     - embedding_file_suffix: the suffix of the embedding files
#     - meta_file_suffix: the suffix of the meta files
#     - embedding_key: the key to the embedding in the embedding file
#     - meta_key: the key to the meta in the meta file
#     - gpu: whether to use gpu for building the index
#     - num_workers: the number of threads to use for building the index
#     - index_desc: the type of the index to build, different index may suits for fast or memory efficient building
#     - output_dir: the directory to save the index

import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import faiss
import h5py
import numpy as np
import scanpy as sc
from tqdm import tqdm

PathLike = Union[str, os.PathLike]


class FaissIndexBuilder:
    """
    Build index for similarity search using faiss.
    """

    def __init__(
        self,
        embedding_dir: PathLike,
        output_dir: PathLike,
        meta_dir: Optional[PathLike] = None,
        recursive: bool = True,
        embedding_file_suffix: str = ".h5ad",
        meta_file_suffix: Optional[str] = None,
        embedding_key: Optional[str] = None,
        meta_key: Optional[str] = "cell_type",
        gpu: bool = False,
        num_workers: Optional[int] = None,
        index_desc: str = "PCA64,IVF16384_HNSW32,PQ16",
    ):
        """
        Initialize an AtlasIndexBuilder object.

        Args:
            embedding_dir (PathLike): Path to the directory containing the input embeddings.
            output_dir (PathLike): Path to the directory where the index files will be saved.
            meta_dir (PathLike, optional): Path to the directory containing the metadata files. If None, the metadata will be loaded from the embedding files. Defaults to None.
            recursive (bool): Whether to search the embedding and meta directory recursively. Defaults to True.
            embedding_file_suffix (str, optional): Suffix of the embedding files. Defaults to ".h5ad" in AnnData format.
            meta_file_suffix (str, optional): Suffix of the metadata files. Defaults to None.
            embedding_key (str, optional): Key to access the embeddings in the input files. If None, will require the input files to be in AnnData format and use the X field. Defaults to None.
            meta_key (str, optional): Key to access the metadata in the input files. Defaults to "cell_type".
            gpu (bool): Whether to use GPU acceleration. Defaults to False.
            num_workers (int, optional): Number of threads to use for CPU parallelism. If None, will use all available cores. Defaults to None.
            index_desc (str, optional): Faiss index factory str, see [here](https://github.com/facebookresearch/faiss/wiki/The-index-factory) and [here](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index#if-1m---10m-ivf65536_hnsw32). Defaults to "PCA64,IVF16384_HNSW32,PQ16".
        """
        self.embedding_dir = embedding_dir
        self.output_dir = output_dir
        self.meta_dir = meta_dir
        self.recursive = recursive
        self.embedding_file_suffix = embedding_file_suffix
        self.meta_file_suffix = meta_file_suffix
        self.embedding_key = embedding_key
        self.meta_key = meta_key
        self.gpu = gpu
        self.num_workers = num_workers
        self.index_desc = index_desc

        if self.num_workers is None:
            try:
                self.num_workers = len(os.sched_getaffinity(0))
                print("Number of available cores: {}".format(self.num_workers))
            except Exception:
                self.num_workers = min(10, os.cpu_count())

        if self.meta_dir is None:  # metadata and embeddings are in the same file
            self.META_FROM_EMBEDDING = True

        if self.embedding_key is None:
            if embedding_file_suffix != ".h5ad":
                raise ValueError(
                    "embedding_key is required when embedding_file_suffix is not .h5ad"
                )

        # See the index factory https://github.com/facebookresearch/faiss/wiki/Lower-memory-footprint#simplifying-index-construction
        # Choose index https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index, particularly, see these options:
        #     - https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index#if-quite-important-then-opqm_dpqmx4fsr
        #     - https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index#if-quite-important-then-opqm_dpqmx4fsr
        #     - For the clustering option, see https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index#if-quite-important-then-opqm_dpqmx4fsr and https://gist.github.com/mdouze/46d6bbbaabca0b9778fca37ed2bcccf6
        # May choose the index option based on the benchmark here https://github.com/facebookresearch/faiss/wiki/Indexing-1G-vectors#10m-datasets

    def _load_data(self):
        # Load embeddings and meta labels
        embeddings = []
        meta_labels = []
        if self.META_FROM_EMBEDDING and self.embedding_file_suffix == ".h5ad":
            embedding_files = (
                list(Path(self.embedding_dir).rglob("*" + self.embedding_file_suffix))
                if self.recursive
                else list(
                    Path(self.embedding_dir).glob("*" + self.embedding_file_suffix)
                )
            )
            embedding_files = [str(f) for f in embedding_files]
            embedding_files = sorted(embedding_files)

            if self.num_workers > 1:
                raise NotImplementedError
            else:
                for file in tqdm(
                    embedding_files, desc="Loading embeddings and metalabels"
                ):
                    adata = sc.read(file)
                    # TODO: set the embedding_key according to self.embedding_key
                    embedding = adata.X.astype(np.float32)
                    if not isinstance(embedding, np.ndarray):
                        embedding = embedding.toarray().astype(np.float32)
                    meta_label = adata.obs[self.meta_key].values
                    embeddings.append(embedding)
                    meta_labels.append(meta_label)
                    del adata
        else:
            raise NotImplementedError
        embeddings = np.concatenate(embeddings, axis=0, dtype=np.float32)
        meta_labels = np.concatenate(meta_labels, axis=0)

        assert embeddings.shape[0] == meta_labels.shape[0]

        return embeddings, meta_labels

    def build_index(self) -> Tuple[faiss.Index, np.ndarray]:
        # Load embeddings and meta labels
        embeddings, meta_labels = self._load_data()

        # Build index
        index = faiss.index_factory(
            embeddings.shape[1], self.index_desc, faiss.METRIC_L2
        )
        nprobe = _auto_set_nprobe(index)
        if self.gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        index.verbose = True
        print(
            f"Training index {self.index_desc} on {embeddings.shape[0]} embeddings ..."
        )
        index.train(embeddings)
        print("Adding embeddings to index ...")
        index.add(embeddings)

        # Save index
        os.makedirs(self.output_dir, exist_ok=True)
        # create sub folder if the output_dir is not empty, and throw warning
        if len(os.listdir(self.output_dir)) > 0:
            print(
                f"Warning: the output_dir {self.output_dir} is not empty, the index will be saved to a sub folder named index"
            )
            self.output_dir = os.path.join(self.output_dir, "index")
            assert not os.path.exists(self.output_dir)
            os.makedirs(self.output_dir, exist_ok=True)

        # save index file, meta file and a json of index config params
        index_file = os.path.join(self.output_dir, "index.faiss")
        meta_file = os.path.join(self.output_dir, "meta.h5ad")
        index_config_file = os.path.join(self.output_dir, "index_config.json")
        faiss.write_index(
            faiss.index_gpu_to_cpu(index) if self.gpu else index, index_file
        )
        with h5py.File(meta_file, "w") as f:
            f.create_dataset(
                "meta_labels", data=meta_labels, compression="gzip", chunks=True
            )
        with open(index_config_file, "w") as f:
            json.dump(
                {
                    "embedding_dir": self.embedding_dir,
                    "meta_dir": self.meta_dir,
                    "recursive": self.recursive,
                    "embedding_file_suffix": self.embedding_file_suffix,
                    "meta_file_suffix": self.meta_file_suffix,
                    "embedding_key": self.embedding_key,
                    "meta_key": self.meta_key,
                    "gpu": self.gpu,
                    "num_workers": self.num_workers,
                    "index_desc": self.index_desc,
                    "num_embeddings": embeddings.shape[0],
                    "num_features": embeddings.shape[1],
                    "nprobe": nprobe,
                },
                f,
            )
        print(f"All files saved to {self.output_dir}")
        print(
            f"Index saved to {index_file}, "
            f"file size: {os.path.getsize(index_file) / 1024 / 1024} MB"
        )

        return index, meta_labels

    def load_index(self) -> Tuple[faiss.Index, np.ndarray]:
        """
        Load the index from self.output_dir.

        Returns:
            faiss.Index: The loaded index and meta labels.
        """
        return load_index(self.output_dir, use_config_file=False, use_gpu=self.gpu)


def load_index(
    index_dir: PathLike,
    use_config_file=True,
    use_gpu=False,
    nprobe=None,
) -> Tuple[faiss.Index, np.ndarray]:
    """
    Load index from disk.

    Args:
        index_dir (PathLike): Path to the directory containing the index files.
        use_config_file (bool, optional): Whether to load the index config file. If True, will load the index config file and use the parameters of gpu, nprobe. Defaults to True.
        use_gpu (bool, optional): Whether to use GPU acceleration. Only used when use_config_file is False. Defaults to False.
        nprobe (int, optional): The nprobe to set if index contains :class:`faiss.IndexIVF`. If None, will set based on the number of clusters. Only used when use_config_file is False. Defaults to None.

    Returns:
        faiss.Index: The loaded index and meta labels.
    """

    index_file = os.path.join(index_dir, "index.faiss")
    meta_file = os.path.join(index_dir, "meta.h5ad")
    index_config_file = os.path.join(index_dir, "index_config.json")

    print(f"Loading index and meta from {index_dir} ...")
    index = faiss.read_index(index_file)
    with h5py.File(meta_file, "r") as f:
        meta_labels = f["meta_labels"][:]
    print(f"Index loaded, num_embeddings: {index.ntotal}")
    if use_config_file:
        with open(index_config_file, "r") as f:
            config = json.load(f)
        use_gpu = config["gpu"]
        nprobe = config["nprobe"]

    _auto_set_nprobe(index, nprobe=nprobe)

    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    return index, meta_labels


def _auto_set_nprobe(index: faiss.Index, nprobe: int = None) -> Optional[int]:
    """
    Set nprobe for IVF index based on the number of clusters.

    Args:
        index (faiss.Index): The index to set nprobe.
        nprobe (int, optional): The nprobe to set. If None, will set based on the number of clusters. Defaults to None.

    Returns:
        int: The nprobe set.
    """

    # set nprobe if IVF index
    index_ivf = faiss.try_extract_index_ivf(index)
    if index_ivf:
        nlist = index_ivf.nlist
        ori_nprobe = index_ivf.nprobe
        index_ivf.nprobe = (
            nprobe
            if nprobe is not None
            else 16
            if nlist <= 1e3
            else 32
            if nlist <= 4e3
            else 64
            if nlist <= 1.6e4
            else 128
        )
        print(
            f"Set nprobe from {ori_nprobe} to {index_ivf.nprobe} for {nlist} clusters"
        )
        return index_ivf.nprobe


if __name__ == "__main__":
    # Set options
    embedding_dir = "/scratch/ssd004/datasets/cellxgene/embed/"
    output_dir = "/scratch/hdd001/home/haotian/projects/cellxemb/all"
    embedding_file_suffix = ".h5ad"
    gpu = True
    index_desc = "PCA64,IVF16384_HNSW32,PQ16"
    num_workers = 1

    # Build index
    builder = FaissIndexBuilder(
        embedding_dir,
        output_dir=output_dir,
        embedding_file_suffix=embedding_file_suffix,
        gpu=gpu,
        num_workers=num_workers,
        index_desc=index_desc,
    )
    index, meta_labels = builder.build_index()
