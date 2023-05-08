# scGPT

This is the official codebase for **scGPT: Towards Building a Foundation Model for Single-Cell Multi-omics Using Generative AI**.

## Installation

scGPT is available on PyPI. To install scGPT, run the following command:

```bash
$ pip install scgpt
```

[Optional] We recommend using [wandb](https://wandb.ai/) for logging and visualization.

```bash
$ pip install wandb
```

For developing, we are using the [Poetry](https://python-poetry.org/) package manager. To install Poetry, follow the instructions [here](https://python-poetry.org/docs/#installation).

```bash
$ git clone this-repo-url
$ cd scGPT
$ poetry install
```

**Note**: The `flash-attn` dependency usually requires specific GPU and CUDA version. If you encounter any issues, please refer to the [flash-attn](https://github.com/HazyResearch/flash-attention/tree/main) repository for installation instructions.

## Pretrained scGPT checkpoints

Please download the pretrained scGPT checkpoints from [here](https://drive.google.com/drive/folders/1kkug5C7NjvXIwQGGaGoqXTk_Lb_pDrBU?usp=sharing).

## Fine-tune scGPT for scRNA-seq integration

Please see our example code in [examples/finetune_integration.py](examples/finetune_integration.py). By default, the script assumes the scGPT checkpoint folder stored in the `examples/save` directory.

## To-do-list

- [x] Upload the pretrained model checkpoint
- [x] Publish to pypi
- [ ] Provide the pretraining code with generative attention masking
- [ ] Finetuning examples for multi-omics integration, cell tyep annotation, perturbation prediction, cell generation
- [ ] Example code for Gene Regulatory Network analysis
- [ ] Documentation website with readthedocs
- [ ] Bump up to pytorch 2.0
- [ ] New pretraining on larger datasets
- [ ] Reference mapping example

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
author={Cui, Haotian and Wang, Chloe and Maan, Hassaan and Wang, Bo},
journal={bioRxiv},
year={2023},
publisher={Cold Spring Harbor Laboratory}
}
```
