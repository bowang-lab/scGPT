Introduction
============

Welcome to the documentation for scGPT, a Python package for single-cell multi-omic data analysis using pretrained foundation models. This package is based on the work of scGPT (`GitHub <https://github.com/bowang-lab/scGPT>`_, `preprint <https://www.biorxiv.org/content/10.1101/2023.04.30.538439>`_) and provides a set of functions for data preprocessing, visualization, and model evaluation.

Our package is built on the foundation of the scGPT model, which is the first single-cell foundation model built through generative pre-training on over 33 million cells. The scGPT model incorporates innovative techniques to overcome methodology and engineering challenges specific to pre-training on large-scale single-cell omic data. By adapting the transformer architecture, we enable the simultaneous learning of cell and gene representations, facilitating a comprehensive understanding of cellular characteristics based on gene expression.

Our package provides a set of functions for data preprocessing, visualization, and model evaluation that are compatible with the scikit-learn library. These functions enable users to preprocess their single-cell RNA-seq data, visualize the results, and evaluate the performance of the scGPT model on downstream tasks such as multi-batch integration, multi-omic integration, cell-type annotation, genetic perturbation prediction, and gene network inference.

In addition to the functions provided by our package, we also provide a tutorial that demonstrates how to use the scGPT model for single-cell RNA-seq data analysis. This tutorial includes step-by-step instructions for preprocessing the data, training the scGPT model, and evaluating its performance on downstream tasks.