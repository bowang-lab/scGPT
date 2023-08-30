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
#     - index_param: the parameters for the index
#     - pca: whether to use pca to reduce the dimension of the embedding
#     - output_dir: the directory to save the index

import os
import pickle
import threading
from pathlib import Path

import faiss
import numpy as np
import scanpy as sc
from tqdm import tqdm


class FaissIndexBuilder:
    def __init__(
        self,
        embedding_dir,
        meta_dir,
        embedding_file_suffix,
        meta_file_suffix,
        embedding_key,
        meta_key,
        gpu,
        num_threads,
        index,
        index_param,
        pca,
        output_dir,
    ):
        self.embedding_dir = embedding_dir
        self.meta_dir = meta_dir
        self.embedding_file_suffix = embedding_file_suffix
        self.meta_file_suffix = meta_file_suffix
        self.embedding_key = embedding_key
        self.meta_key = meta_key
        self.gpu = gpu
        self.num_threads = num_threads
        self.index = index
        self.index_param = index_param
        self.pca = pca
        self.output_dir = output_dir

    def build_index(self):
        # Load embeddings and meta labels
        embeddings = []
        meta_labels = []
        for embedding_file, meta_file in zip(
            sorted(Path(self.embedding_dir).glob(f"*{self.embedding_file_suffix}")),
            sorted(Path(self.meta_dir).glob(f"*{self.meta_file_suffix}")),
        ):
            embedding = np.load(embedding_file)[self.embedding_key]
            meta_label = np.load(meta_file)[self.meta_key]
            embeddings.append(embedding)
            meta_labels.append(meta_label)
        embeddings = np.concatenate(embeddings, axis=0)
        meta_labels = np.concatenate(meta_labels, axis=0)

        # Reduce dimension using PCA
        if self.pca:
            pca = faiss.PCAMatrix(embeddings.shape[1], embeddings.shape[1] // 2)
            pca.train(embeddings)
            embeddings = pca.apply_py(embeddings)

        # Build index
        if self.gpu:
            res = faiss.StandardGpuResources()
            index = getattr(faiss, self.index)(embeddings.shape[1], **self.index_param)
            index = faiss.index_cpu_to_gpu(res, 0, index)
        else:
            index = getattr(faiss, self.index)(embeddings.shape[1], **self.index_param)
        index.train(embeddings)
        index.add(embeddings)

        # Save index
        os.makedirs(self.output_dir, exist_ok=True)
        index_file = os.path.join(self.output_dir, "index")
        meta_file = os.path.join(self.output_dir, "meta.pkl")
        faiss.write_index(index, index_file)
        with open(meta_file, "wb") as f:
            pickle.dump(meta_labels, f)


if __name__ == "__main__":
    # Set options
    embedding_dir = "path/to/embedding/dir"
    meta_dir = "path/to/meta/dir"
    embedding_file_suffix = ".npy"
    meta_file_suffix = ".npy"
    embedding_key = "embedding"
    meta_key = "meta"
    gpu = True
    num_threads = 4
    index = "IVF4096,Flat"
    index_param = {"nlist": 4096, "nprobe": 32}
    pca = True
    output_dir = "path/to/output/dir"

    # Build index
    builder = FaissIndexBuilder(
        embedding_dir,
        meta_dir,
        embedding_file_suffix,
        meta_file_suffix,
        embedding_key,
        meta_key,
        gpu,
        num_threads,
        index,
        index_param,
        pca,
        output_dir,
    )
    builder.build_index()
