import collections
import operator
import os
from typing import Mapping

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


# The class is modified from https://github.com/nceglia/genevector
class GeneEmbedding(object):
    def __init__(self, embeddings: Mapping):
        self.embeddings = embeddings
        # self.context = dataset.data
        self.vector = []
        self.genes = []
        for gene in tqdm.tqdm(self.embeddings.keys()):
            self.vector.append(self.embeddings[gene])
            self.genes.append(gene)

    def read_embedding(self, filename):
        embedding = dict()
        lines = open(filename, "r").read().splitlines()[1:]
        for line in lines:
            vector = line.split()
            gene = vector.pop(0)
            embedding[gene] = np.array([float(x) for x in vector])
        return embedding

    def get_adata(self, resolution=20):
        mat = np.array(self.vector)
        np.savetxt(".tmp.txt", mat)
        gdata = sc.read_text(".tmp.txt")
        os.remove(".tmp.txt")
        gdata.obs.index = self.genes
        sc.pp.neighbors(gdata, use_rep="X")
        sc.tl.leiden(gdata, resolution=resolution)
        sc.tl.umap(gdata)
        return gdata

    def plot_similarities(self, gene, n_genes=10, save=None):
        df = self.compute_similarities(gene).head(n_genes)
        _, ax = plt.subplots(1, 1, figsize=(3, 6))
        sns.barplot(data=df, y="Gene", x="Similarity", palette="magma_r", ax=ax)
        ax.set_title("{} Similarity".format(gene))
        if save != None:
            plt.savefig(save)

    def plot_metagene(self, gdata, mg=None, title="Gene Embedding"):
        highlight = []
        labels = []
        clusters = collections.defaultdict(list)
        for x, y in zip(gdata.obs["leiden"], gdata.obs.index):
            clusters[x].append(y)
            if x == mg:
                highlight.append(str(x))
                labels.append(y)
            else:
                highlight.append("_Other")
        _labels = []
        for gene in labels:
            _labels.append(gene)
        gdata.obs["Metagene {}".format(mg)] = highlight
        _, ax = plt.subplots(1, 1, figsize=(8, 6))
        sc.pl.umap(gdata, alpha=0.5, show=False, size=100, ax=ax)
        sub = gdata[gdata.obs["Metagene {}".format(mg)] != "_Other"]
        sc.pl.umap(
            sub,
            color="Metagene {}".format(mg),
            title=title,
            size=200,
            show=False,
            add_outline=False,
            ax=ax,
        )
        for gene, pos in zip(gdata.obs.index, gdata.obsm["X_umap"].tolist()):
            if gene in _labels:
                ax.text(
                    pos[0] + 0.04,
                    pos[1],
                    str(gene),
                    fontsize=6,
                    alpha=0.9,
                    fontweight="bold",
                )
        plt.tight_layout()

    def plot_metagenes_scores(self, adata, metagenes, column, plot=None):
        plt.figure(figsize=(5, 13))
        matrix = []
        meta_genes = []
        cfnum = 1
        cfams = dict()
        for cluster, vector in metagenes.items():
            row = []
            cts = []
            for ct in set(adata.obs[column]):
                sub = adata[adata.obs[column] == ct]
                val = np.mean(sub.obs[str(cluster) + "_SCORE"].tolist())
                row.append(val)
                cts.append(ct)
            matrix.append(row)
            label = str(cluster) + "_SCORE: " + ", ".join(vector[:10])
            if len(set(vector)) > 10:
                label += "*"
            meta_genes.append(label)
            cfams[cluster] = label
            cfnum += 1
        matrix = np.array(matrix)
        df = pd.DataFrame(matrix, index=meta_genes, columns=cts)
        plt.figure()
        sns.clustermap(
            df,
            figsize=(5, 9),
            dendrogram_ratio=0.1,
            cmap="mako",
            yticklabels=True,
            standard_scale=0,
        )
        plt.tight_layout()
        if plot:
            plt.savefig(plot)

    def score_metagenes(self, adata, metagenes):
        for p, genes in metagenes.items():
            try:
                sc.tl.score_genes(adata, score_name=str(p) + "_SCORE", gene_list=genes)
                scores = np.array(adata.obs[str(p) + "_SCORE"].tolist()).reshape(-1, 1)
                scaler = MinMaxScaler()
                scores = scaler.fit_transform(scores)
                scores = list(scores.reshape(1, -1))[0]
                adata.obs[str(p) + "_SCORE"] = scores
            except Exception as e:
                adata.obs[str(p) + "_SCORE"] = 0.0

    def get_metagenes(self, gdata):
        metagenes = collections.defaultdict(list)
        for x, y in zip(gdata.obs["leiden"], gdata.obs.index):
            metagenes[x].append(y)
        return metagenes

    def compute_similarities(self, gene, subset=None, feature_type=None):
        if gene not in self.embeddings:
            return None
        # if feature_type:
        #     subset = []
        #     for gene in list(self.embeddings.keys()):
        #         if feature_type == self.context.feature_types[gene]:
        #             subset.append(gene)
        embedding = self.embeddings[gene]
        distances = dict()
        if subset:
            targets = set(list(self.embeddings.keys())).intersection(set(subset))
        else:
            targets = list(self.embeddings.keys())
        for target in targets:
            if target not in self.embeddings:
                continue
            v = self.embeddings[target]
            distance = float(
                cosine_similarity(
                    np.array(embedding).reshape(1, -1), np.array(v).reshape(1, -1)
                )[0]
            )
            distances[target] = distance
        sorted_distances = list(
            reversed(sorted(distances.items(), key=operator.itemgetter(1)))
        )
        genes = [x[0] for x in sorted_distances]
        distance = [x[1] for x in sorted_distances]
        df = pd.DataFrame.from_dict({"Gene": genes, "Similarity": distance})
        return df

    # def clusters(self, clusters):
    #     average_vector = dict()
    #     gene_to_cluster = collections.defaultdict(list)
    #     matrix = collections.defaultdict(list)
    #     total_average_vector = []
    #     for gene, cluster in zip(self.context.expressed_genes, clusters):
    #         if gene in self.embeddings:
    #             matrix[cluster].append(self.embeddings[gene])
    #             gene_to_cluster[cluster].append(gene)
    #             total_average_vector.append(self.embeddings[gene])
    #     self.total_average_vector = list(np.average(total_average_vector, axis=0))
    #     for cluster, vectors in matrix.items():
    #         xvec = list(np.average(vectors, axis=0))
    #         average_vector[cluster] = np.subtract(xvec, self.total_average_vector)
    #     return average_vector, gene_to_cluster

    def generate_weighted_vector(self, genes, markers, weights):
        vector = []
        for gene, vec in zip(self.genes, self.vector):
            if gene in genes and gene in weights:
                vector.append(weights[gene] * np.array(vec))
            if gene not in genes and gene in markers and gene in weights:
                vector.append(list(weights[gene] * np.negative(np.array(vec))))
        return list(np.sum(vector, axis=0))

    def generate_vector(self, genes):
        vector = []
        for gene, vec in zip(self.genes, self.vector):
            if gene in genes:
                vector.append(vec)
        assert len(vector) != 0, genes
        return list(np.average(vector, axis=0))

    def generate_weighted_vector(self, genes, weights):
        vector = []
        weight = []
        for gene, vec in zip(self.genes, self.vector):
            if gene in genes and gene in weights:
                vector.append(vec)
                weight.append(weights[gene])
        assert len(vector) != 0, genes
        return list(np.average(vector, axis=0, weights=weight))

    def cluster_definitions_as_df(self, top_n=20):
        similarities = self.cluster_definitions
        clusters = []
        symbols = []
        for key, genes in similarities.items():
            clusters.append(key)
            symbols.append(", ".join(genes[:top_n]))
        df = pd.DataFrame.from_dict({"Cluster Name": clusters, "Top Genes": symbols})
        return df

    @staticmethod
    def read_vector(vec):
        lines = open(vec, "r").read().splitlines()
        dims = lines.pop(0)
        vecs = dict()
        for line in lines:
            try:
                line = line.split()
                gene = line.pop(0)
                vecs[gene] = list(map(float, line))
            except Exception as e:
                continue
        return vecs, dims

    def get_similar_genes(self, vector):
        distances = dict()
        targets = list(self.embeddings.keys())
        for target in targets:
            if target not in self.embeddings:
                continue
            v = self.embeddings[target]
            distance = float(
                cosine_similarity(
                    np.array(vector).reshape(1, -1), np.array(v).reshape(1, -1)
                )[0]
            )
            distances[target] = distance
        sorted_distances = list(
            reversed(sorted(distances.items(), key=operator.itemgetter(1)))
        )
        genes = [x[0] for x in sorted_distances]
        distance = [x[1] for x in sorted_distances]
        df = pd.DataFrame.from_dict({"Gene": genes, "Similarity": distance})
        return df

    def generate_network(self, threshold=0.5):
        G = nx.Graph()
        a = pd.DataFrame.from_dict(self.embeddings).to_numpy()
        similarities = cosine_similarity(a.T)
        genes = list(self.embeddings.keys())
        similarities[similarities < threshold] = 0
        edges = []
        nz = list(zip(*similarities.nonzero()))
        for n in tqdm.tqdm(nz):
            edges.append((genes[n[0]], genes[n[1]]))
        G.add_nodes_from(genes)
        G.add_edges_from(edges)
        return G

    @staticmethod
    def average_vector_results(vec1, vec2, fname):
        output = open(fname, "w")
        vec1, dims = GeneEmbedding.read_vector(vec1)
        vec2, _ = GeneEmbedding.read_vector(vec2)
        genes = list(vec1.keys())
        output.write(dims + "\n")
        for gene in genes:
            v1 = vec1[gene]
            v2 = vec2[gene]
            meanv = []
            for x, y in zip(v1, v2):
                meanv.append(str((x + y) / 2))
            output.write("{} {}\n".format(gene, " ".join(meanv)))
        output.close()
