# This script is used to build the index for similarity search using faisss. It supports loading embeddings and meta labels from directories and add to faiss index. The index is saved to disk for later use.
# Options can be set to custom the building process, including:
#     - embedding_dir: the directory to the embedding files
#     - meta_dir: the directory to the meta files
#     - embedding_file_suffix: the suffix of the embedding files
#     - meta_file_suffix: the suffix of the meta files
#     - embedding_key: the key to the embedding in the embedding file
#     - meta_key: the key to the meta in the meta file
#     - gpu: whether to use gpu for building the index
#     - num_threads: the number of threads to use for building the index
#     - index: the type of the index to build, different index may suits for fast or memory efficient building
#     - output_dir: the directory to save the index

import json
import os
import pickle
import threading
from pathlib import Path
from typing import Dict, Optional, Union

import faiss
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
        recusive: bool = True,
        embedding_file_suffix: str = ".h5ad",
        meta_file_suffix: Optional[str] = None,
        embedding_key: Optional[str] = None,
        meta_key: Optional[str] = "cell_type",
        gpu: bool = False,
        num_threads: Optional[int] = None,
        index: str = "PCA64,IVF16384_HNSW32,PQ16",
    ):
        """
        Initialize an AtlasIndexBuilder object.

        Args:
            embedding_dir (PathLike): Path to the directory containing the input embeddings.
            output_dir (PathLike): Path to the directory where the index files will be saved.
            meta_dir (PathLike, optional): Path to the directory containing the metadata files. If None, the metadata will be loaded from the embedding files. Defaults to None.
            recusive (bool): Whether to search the embedding and meta directory recursively. Defaults to True.
            embedding_file_suffix (str, optional): Suffix of the embedding files. Defaults to ".h5ad" in AnnData format.
            meta_file_suffix (str, optional): Suffix of the metadata files. Defaults to None.
            embedding_key (str, optional): Key to access the embeddings in the input files. If None, will require the input files to be in AnnData format and use the X field. Defaults to None.
            meta_key (str, optional): Key to access the metadata in the input files. Defaults to "cell_type".
            gpu (bool): Whether to use GPU acceleration. Defaults to False.
            num_threads (int, optional): Number of threads to use for CPU parallelism. If None, will use all available cores. Defaults to None.
            index (str, optional): Faiss index factory str, see [here](https://github.com/facebookresearch/faiss/wiki/The-index-factory) and [here](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index#if-1m---10m-ivf65536_hnsw32). Defaults to "PCA64,IVF16384_HNSW32,PQ16".
        """
        self.embedding_dir = embedding_dir
        self.output_dir = output_dir
        self.meta_dir = meta_dir
        self.recusive = recusive
        self.embedding_file_suffix = embedding_file_suffix
        self.meta_file_suffix = meta_file_suffix
        self.embedding_key = embedding_key
        self.meta_key = meta_key
        self.gpu = gpu
        self.num_threads = num_threads
        self.index = index

        if self.num_threads is None:
            try:
                self.num_threads = len(os.sched_getaffinity(0))
                print("Number of available cores: {}".format(self.num_threads))
            except Exception:
                self.num_threads = min(10, os.cpu_count())

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
                if self.recusive
                else os.listdir(self.embedding_dir)
            )
            embedding_files = [
                f for f in embedding_files if f.endswith(self.embedding_file_suffix)
            ]
            embedding_files = sorted(embedding_files)
            if self.num_threads > 1:
                lock = threading.Lock()
                with tqdm(
                    total=len(embedding_files), desc="Loading embeddings and metalabels"
                ) as pbar:

                    def _load_embedding(file):
                        embedding = sc.read(file)
                        embedding = embedding.X
                        meta_label = embedding.obs[self.meta_key].values
                        with lock:
                            embeddings.append(embedding)
                            meta_labels.append(meta_label)
                            pbar.update(1)

                    threads = []
                    for file in embedding_files:
                        thread = threading.Thread(target=_load_embedding, args=(file,))
                        thread.start()
                        threads.append(thread)
                    for thread in threads:
                        thread.join()
            else:
                for file in tqdm(
                    embedding_files, desc="Loading embeddings and metalabels"
                ):
                    embedding = sc.read(file)
                    embedding = embedding.X
                    meta_label = embedding.obs[self.meta_key].values
                    embeddings.append(embedding)
                    meta_labels.append(meta_label)
        else:
            raise NotImplementedError
        embeddings = np.concatenate(embeddings, axis=0, dtype=np.float32)
        meta_labels = np.concatenate(meta_labels, axis=0)

        assert embeddings.shape[0] == meta_labels.shape[0]

        return embeddings, meta_labels

    def _auto_set_nprobe(self, index: faiss.Index):
        # set nprobe if IVF index
        index_ivf = faiss.try_extract_index_ivf(index)
        if index_ivf:
            nlist = index_ivf.nlist
            ori_nprobe = index_ivf.nprobe
            index_ivf.nprobe = (
                16
                if nlist <= 1e4
                else 32
                if nlist <= 8e4
                else 64
                if nlist <= 4e5
                else 128
            )
            print(
                f"Set nprobe from {ori_nprobe} to {index_ivf.nprobe} for {nlist} clusters"
            )

    def build_index(self) -> faiss.Index:
        # Load embeddings and meta labels
        embeddings, meta_labels = self._load_data()

        # Build index
        index = faiss.index_factory(embeddings.shape[1], self.index, faiss.METRIC_L2)
        self._auto_set_nprobe(index)
        if self.gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        index.verbose = True
        print(f"Training index {self.index} on {embeddings.shape[0]} embeddings ...")
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
        meta_file = os.path.join(self.output_dir, "meta.pkl")
        index_config_file = os.path.join(self.output_dir, "index_config.json")
        faiss.write_index(
            faiss.index_gpu_to_cpu(index) if self.gpu else index, index_file
        )
        with open(meta_file, "wb") as f:
            pickle.dump(meta_labels, f)
        with open(index_config_file, "w") as f:
            json.dump(
                {
                    "embedding_dir": self.embedding_dir,
                    "meta_dir": self.meta_dir,
                    "recusive": self.recusive,
                    "embedding_file_suffix": self.embedding_file_suffix,
                    "meta_file_suffix": self.meta_file_suffix,
                    "embedding_key": self.embedding_key,
                    "meta_key": self.meta_key,
                    "gpu": self.gpu,
                    "num_threads": self.num_threads,
                    "index": self.index,
                    "num_embeddings": embeddings.shape[0],
                    "num_features": embeddings.shape[1],
                },
                f,
            )
        print(f"All files saved to {self.output_dir}")
        print(
            f"Index saved to {index_file}, "
            f"file size: {os.path.getsize(index_file) / 1024 / 1024} MB"
        )

        return index

    # TODO: load index
    # TODO: set nprobe


if __name__ == "__main__":
    # Set options
    embedding_dir = "path/to/embedding/dir"
    meta_dir = "path/to/meta/dir"
    embedding_file_suffix = ".npy"
    meta_file_suffix = ".npy"
    embedding_key = "embedding"
    meta_key = "meta"
    gpu = False
    index = "PCA64,IVF16384_HNSW32,PQ16"
    output_dir = "path/to/output/dir"

    # Build index
    builder = FaissIndexBuilder(
        embedding_dir,
        meta_dir,
        embedding_file_suffix=embedding_file_suffix,
        meta_file_suffix=meta_file_suffix,
        embedding_key=embedding_key,
        meta_key=meta_key,
        gpu=gpu,
        index=index,
        output_dir=output_dir,
    )
    builder.build_index()