# Zero-Shot Applications

This directory contains tutorials for zero-shot applications of scGPT. We provide instructions, notebooks of best practices, and we discuss notable findings related to the zero-shot application settings.

## Best practice and results

We provide three notebooks:
| Description | Notebook | Colab |
| :---------- | :--- | :---- |
| Zero-shot reference mapping | [here](Tutorial_ZeroShot_Reference_Mapping.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bowang-lab/scGPT/blob/main/tutorials/zero-shot/Tutorial_ZeroShot_Reference_Mapping.ipynb) |
| Zero-shot integration | [here](Tutorial_ZeroShot_Integration.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bowang-lab/scGPT/blob/main/tutorials/zero-shot/Tutorial_ZeroShot_Integration.ipynb) |
| Zero-shot integration with continual pretrained model | [here](Tutorial_ZeroShot_Integration_Continual_Pretraining.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bowang-lab/scGPT/blob/main/tutorials/zero-shot/Tutorial_ZeroShot_Integration_Continual_Pretraining.ipynb) |

In the reference mapping and integration notebooks, we provide a step-by-step guide for running the zero-shot applications. Running with zero-shot manner is super fast and demonstrating great performance on multiple datasets.

## New continual pretrained model for cell embedding related tasks

In the third notebook, [Tutorial_ZeroShot_Integration_Continual_Pretraining.ipynb](Tutorial_ZeroShot_Integration_Continual_Pretraining.ipynb), we provide a new continual pretrained model for cell embedding related tasks. The model, [scGPT_CP](https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y), inherits the pre-trained scGPT whole-human model checkpoint, and is further supervised by extra cell type labels (using the [Tabula Sapiens](https://tabula-sapiens-portal.ds.czbiohub.org/) dataset) during the continual pre-training stage. It can achieve comparable or better zero-shot performance on cell embedding related tasks compared to the original checkpoint, especially on datasets with observable technical batch effects.

<!-- ## Analyzing influence of parameters -->

## Challenges and potential limitations

Great zero-shot performance for reference mapping and integration were observed in the provided notebooks. In the same time, it is worth mentioning that the current pre-training is a purely data-driven and self-supervised procedure that optimizes the model to reconstruct gene expressions in sequenced samples. This procedure offers the benefits of avoiding potential human bias in the meta-labels and treating all raw signals in the data "equally". However, it inherently omits other requirements, for example it does not inherently include the mitigation of the technical variations introduced by sequencing protocols and devices. For this reason, we first **(1)** provide the newly continual pretrained model, scGPT_CP, which takes cell types and technical batch effects into consideration intrinsically, as demonstrated in [Tutorial_ZeroShot_Integration_Continual_Pretraining.ipynb](Tutorial_ZeroShot_Integration_Continual_Pretraining.ipynb). **(2)** We also provide a discussion on the technical batch effects and their role in the pre-training process as follows.

In single-cell data analysis, technical batch effects indicate data distribution shifts that are driven by the sequencing procedure, rather than biological variations. The mitigation of such effects is instrumental for more accurate visualization and clustering of cell populations across multiple samples. Numerous elements contribute to the introduction of batch effects in the collation of cell atlases, including but not limited to, capturing times, personnel involved, reagent batches, types of equipment, sequencing protocols, and technological platforms. Considering the pretraining process reconstructs binned gene expressions from other observed genes, we can classify these factors into two broad types based on their influence on pretraining: (1) The type I factors are the ones that may largely have global effects on the capturing of all gene expression, such as capturing times or sequencing depth. The pre-training may mitigate these factors through repeated training on large data examples. (2) The type II factors have gene-specific influences. Some of the factors (i.e. sequencing protocols and machines) can alter the gene expression distributions. Rather than being ameliorated, these effects can be faithfully encapsulated during the pre-training process. From the standpoint of our pre-training objectives, these factors are not considered noise but important signals to the gene expression reconstruction task.

We would argue this limitation on the mitigating of type II technical effects is a necessary tradeoff for maintaining an "unbiased" pre-training. While it may be plausible to incorporate a batch effect mitigation objective within the pre-training scheme, the added objective could compromise the original pre-training goals aimed at reconstructing biological gene expression because of the complex, often non-linear relationships between technical and biological factors. Consequently, we advocate maintaining an "unbiased" data-driven approach during pre-training, thereby offering the flexibility to prioritize specific analytical objectives in subsequent fine-tuning stages.
