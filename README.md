# scGPT

This is the official codebase for **scGPT: Towards Building a Foundation Model for Single-Cell Multi-omics Using Generative AI**.

[![Preprint](https://img.shields.io/badge/preprint-available-brightgreen)](https://www.biorxiv.org/content/10.1101/2023.04.30.538439) &nbsp;
[![Documentation](https://img.shields.io/badge/docs-available-brightgreen)](https://scgpt.readthedocs.io/en/latest/) &nbsp;
[![PyPI version](https://badge.fury.io/py/scgpt.svg)](https://badge.fury.io/py/scgpt) &nbsp;
[![Downloads](https://pepy.tech/badge/scgpt)](https://pepy.tech/project/scgpt) &nbsp;
![Webapp](https://img.shields.io/website?url=https%3A%2F%2Fscgpthub.org&up_color=chartreuse%20&logo=gotomeeting&logoColor=%23FFB3FF&label=WebApp&labelColor=%2300CBFF) &nbsp;
[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/username/repo/blob/main/LICENSE)

**!UPDATE**: We have released several new pretrained scGPT checkpoints. Please see the [Pretrained scGPT checkpoints](#pretrained-scGPT-checkpoints) section for more details.

**[2024.02.26]** We have provided a priliminary support for running the pretraining workflow with HuggingFace at the [integrate-huggingface-model](https://github.com/bowang-lab/scGPT/tree/integrate-huggingface-model) branch. We will conduct further testing and merge it to the main branch soon.

**[2023.12.31]** New tutorials about zero-shot applications are now available! Please see find them in the [tutorials/zero-shot](tutorials/zero-shot) directory. We also provide a new continual pretrained model checkpoint for cell embedding related tasks. Please see the [notebook](tutorials/zero-shot/Tutorial_ZeroShot_Integration_Continual_Pretraining.ipynb) for more details.

**[2023.11.07]** As requested by many, now we have made flash-attention an optional dependency. The pretrained weights can be loaded on pytorch CPU, GPU, and flash-attn backends using the same [load_pretrained](https://github.com/bowang-lab/scGPT/blob/f6097112fe5175cd4e221890ed2e2b1815f54010/scgpt/utils/util.py#L304) function, `load_pretrained(target_model, torch.load("path_to_ckpt.pt"))`. An example usage is also [here](https://github.com/bowang-lab/scGPT/blob/f6097112fe5175cd4e221890ed2e2b1815f54010/scgpt/tasks/cell_emb.py#L258).

**[2023.09.05]** We have release a new feature for reference mapping samples to a custom reference dataset or to all the millions of cells collected from CellXGene! With the help of the [faiss](https://github.com/facebookresearch/faiss) library, we achieved a great time and memory efficiency. The index of over 33 millions cells only takes less than 1GB of memory and the similarity search takes less than **1 second for 10,000 query cells on GPU**. Please see the [Reference mapping tutorial](https://github.com/bowang-lab/scGPT/blob/main/tutorials/Tutorial_Reference_Mapping.ipynb) for more details.

### Online apps

scGPT is now available at the following online apps as well, so you can get started simply with your browser!

- Run the [reference mapping app](https://app.superbio.ai/apps/299?id=6548f339a9ed6f6e5560b07d), [cell annotation app](https://app.superbio.ai/apps/274?id=64d205cb980ff714de831ee0) and the [GRN inference app](https://app.superbio.ai/apps/270?id=64b804fb823bc93b64c10a76) with cloud gpus. Thanks to the [Superbio.ai](https://app.superbio.ai/) team for helping create and host the interactive tools.

## Installation

scGPT works with Python >= 3.8 and R >=3.6.1. Please make sure you have the correct version of Python and R installed pre-installation.

scGPT is available on PyPI. To install scGPT, run the following command:

```bash
pip install scgpt "flash-attn<1.0.5"  # optional, recommended
# As of 2023.09, pip install may not run with new versions of the google orbax package, if you encounter related issues, please use the following command instead:
# pip install scgpt "flash-attn<1.0.5" "orbax<0.1.8"
```

[Optional] We recommend using [wandb](https://wandb.ai/) for logging and visualization.

```bash
pip install wandb
```

For developing, we are using the [uv](https://docs.astral.sh/uv/#getting-started) package manager. To install Poetry, follow the instructions [here](https://docs.astral.sh/uv/getting-started/installation).

```bash
$ git clone this-repo-url
$ cd scGPT

# Install without flash-attn
$ uv sync

# Install flash-attn
$ uv sync --extra build # Setup flash-attn build environment
$ uv sync --extra build --extra comile # Install flash-attn

# Activate virtual environment

## Linux/Unix
. .venv/bin/activate

## Windows
. .venv/Scripts/activate
```

**Note**: The `flash-attn` dependency usually requires specific GPU and CUDA version. If you encounter any issues, please refer to the [flash-attn](https://github.com/HazyResearch/flash-attention/tree/main) repository for installation instructions. For now, May 2023, we recommend using CUDA 11.7 and flash-attn<1.0.5 due to various issues reported about installing new versions of flash-attn.

## Pretrained scGPT Model Zoo

Here is the list of pretrained models. Please find the links for downloading the checkpoint folders. We recommend using the `whole-human` model for most applications by default. If your fine-tuning dataset shares similar cell type context with the training data of the organ-specific models, these models can usually demonstrate competitive performance as well. A paired vocabulary file mapping gene names to ids is provided in each checkpoint folder. If ENSEMBL ids are needed, please find the conversion at [gene_info.csv](https://github.com/bowang-lab/scGPT/files/13243634/gene_info.csv).

| Model name                | Description                                             | Download                                                                                     |
| :------------------------ | :------------------------------------------------------ | :------------------------------------------------------------------------------------------- |
| whole-human (recommended) | Pretrained on 33 million normal human cells.            | [link](https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y?usp=sharing) |
| continual pretrained      | For zero-shot cell embedding related tasks.             | [link](https://drive.google.com/drive/folders/1_GROJTzXiAV8HB4imruOTk6PEGuNOcgB?usp=sharing) |
| brain                     | Pretrained on 13.2 million brain cells.                 | [link](https://drive.google.com/drive/folders/1vf1ijfQSk7rGdDGpBntR5bi5g6gNt-Gx?usp=sharing) |
| blood                     | Pretrained on 10.3 million blood and bone marrow cells. | [link](https://drive.google.com/drive/folders/1kkug5C7NjvXIwQGGaGoqXTk_Lb_pDrBU?usp=sharing) |
| heart                     | Pretrained on 1.8 million heart cells                   | [link](https://drive.google.com/drive/folders/1GcgXrd7apn6y4Ze_iSCncskX3UsWPY2r?usp=sharing) |
| lung                      | Pretrained on 2.1 million lung cells                    | [link](https://drive.google.com/drive/folders/16A1DJ30PT6bodt4bWLa4hpS7gbWZQFBG?usp=sharing) |
| kidney                    | Pretrained on 814 thousand kidney cells                 | [link](https://drive.google.com/drive/folders/1S-1AR65DF120kNFpEbWCvRHPhpkGK3kK?usp=sharing) |
| pan-cancer                | Pretrained on 5.7 million cells of various cancer types | [link](https://drive.google.com/drive/folders/13QzLHilYUd0v3HTwa_9n4G4yEF-hdkqa?usp=sharing) |

## Fine-tune scGPT for scRNA-seq integration

Please see our example code in [examples/finetune_integration.py](examples/finetune_integration.py). By default, the script assumes the scGPT checkpoint folder stored in the `examples/save` directory.

## To-do-list

- [x] Upload the pretrained model checkpoint
- [x] Publish to pypi
- [ ] Provide the pretraining code with generative attention masking
- [ ] Finetuning examples for multi-omics integration, cell type annotation, perturbation prediction, cell generation
- [x] Example code for Gene Regulatory Network analysis
- [x] Documentation website with readthedocs
- [x] Bump up to pytorch 2.0
- [x] New pretraining on larger datasets
- [x] Reference mapping example
- [ ] Publish to huggingface model hub

## Contributing

We greatly welcome contributions to scGPT. Please submit a pull request if you have any ideas or bug fixes. We also welcome any issues you encounter while using scGPT.

## Acknowledgements

We sincerely thank the authors of following open-source projects:

- [flash-attention](https://github.com/HazyResearch/flash-attention)
- [scanpy](https://github.com/scverse/scanpy)
- [scvi-tools](https://github.com/scverse/scvi-tools)
- [scib](https://github.com/theislab/scib)
- [datasets](https://github.com/huggingface/datasets)
- [transformers](https://github.com/huggingface/transformers)

## Citing scGPT

```bibtex
@article{cui2023scGPT,
title={scGPT: Towards Building a Foundation Model for Single-Cell Multi-omics Using Generative AI},
author={Cui, Haotian and Wang, Chloe and Maan, Hassaan and Pang, Kuan and Luo, Fengning and Wang, Bo},
journal={bioRxiv},
year={2023},
publisher={Cold Spring Harbor Laboratory}
}
```
