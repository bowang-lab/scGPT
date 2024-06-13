""" Example call for running evaluation
DATASET="path_to/datasets/3faad104-2ab8-4434-816d-474d8d2641db.scb"
DATASET=test
JOB_NAME="cellxgene_3faad1"
LOG_INTERVAL=100
VALID_SIZE_OR_RATIO=0.1
MAX_LENGTH=600
per_proc_batch_size=64
LAYERS=4
MODEL_SCALE=1
python pretrain.py --data-source $DATASET --save-dir ./save/eval-$(date  +%b%d-%H-%M-%Y) --max-seq-len $MAX_LENGTH --batch-size $per_proc_batch_size     --eval-batch-size $(($per_proc_batch_size * 2))     --epochs 3     --lr 0.0001     --warmup-ratio-or-step 0.1     --log-interval $LOG_INTERVAL --trunc-by-sample --no-cls --no-cce --fp16

python pretrain.py \
    --data-source $DATASET \
    --save-dir ./save/eval-$(date +%b%d-%H-%M-%Y) \
    --load-model $CHECKPOINT \
    --max-seq-len $MAX_LENGTH \
    --batch-size $per_proc_batch_size \
    --eval-batch-size $(($per_proc_batch_size * 2)) \
    --epochs 0 \
    --log-interval $LOG_INTERVAL \
    --trunc-by-sample \
    --no-cls \
    --no-cce \
    --fp16

Example call for running fine-tuning
python pretrain.py \
    --data-source $DATASET \
    --save-dir ./save/eval-$(date +%b%d-%H-%M-%Y) \
    --load-model $CHECKPOINT \
    --max-seq-len $MAX_LENGTH \
    --batch-size $per_proc_batch_size \
    --eval-batch-size $(($per_proc_batch_size * 2)) \
    --epochs 3 \
    --lr 0.0001 \
    --warmup-ratio-or-step 0.1 \
    --log-interval $LOG_INTERVAL \
    --trunc-by-sample \
    --no-cls \
    --no-cce \
    --fp16
"""
# %%
import os
import sys
import argparse
import json
import time
from datetime import timedelta
from pathlib import Path
from typing import List, Tuple, Dict, Union, Optional

import scanpy as sc
import numpy as np
import torch
import transformers
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, BatchSampler, RandomSampler, SequentialSampler
from datasets import Dataset, load_dataset, concatenate_datasets


sys.path.insert(0, "../")
import scgpt as scg
from scgpt.model import TransformerModel
from scgpt.loss import masked_mse_loss, masked_relative_error
from scgpt.tokenizer import GeneVocab, random_mask_value
from scgpt.scbank import DataBank
from scgpt.utils import MainProcessOnly
from scgpt import logger


# torch.autograd.set_detect_anomaly(True)

sc.set_figure_params(figsize=(4, 4))
sc.settings.verbosity = "debug"
scg.utils.set_seed(42)

# %%
# argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "-d",
    "--data-source",
    type=str,
    required=True,
    help='The name of the data source (currently support "scvi" datasets), or the '
    "path to the data file.",
)
parser.add_argument(
    "-s",
    "--save-dir",
    type=str,
    required=True,
    help="The directory to save the trained model and the results.",
)
parser.add_argument(
    "--load-model",
    type=str,
    default=None,
    help="The directory containing the model and configs to load and continue training.",
)

# settings for data
parser.add_argument(
    "--n-hvg",
    type=int,
    default=None,
    help="The number of highly variable genes. If set to 0, will use all genes. "
    "Default is None, which will determine the n_hvg automatically.",
)
parser.add_argument(
    "--valid-size-or-ratio",
    type=float,
    default=0.1,
    help="The ratio or size of the validation set size if split the dataset. "
    "If value is between 0 and 1, will be parsed as the ratio. If value is "
    "greater than 1 and be an integer, will be parsed as the size. If value "
    "is 0, will not split the dataset.",
)

parser.add_argument(
    "--grad-accu-steps",
    type=int,
    default=1,
    help="The number of gradient accumulation steps. Default is 1.",
)

# settings for tokenizer
parser.add_argument(
    "--pad-token",
    type=str,
    default="<pad>",
    help="The token to use for padding. Default is <pad>.",
)
parser.add_argument(
    "--input-style",
    type=str,
    choices=["normed_raw", "log1p", "binned"],
    default="binned",
    help="The style of the input data. Default is binned.",
)
parser.add_argument(
    "--input-emb-style",
    type=str,
    choices=["category", "continuous", "scaling"],
    default="continuous",
    help="The style of the input embedding. Default is continuous.",
)
parser.add_argument(
    "--n-bins",
    type=int,
    default=51,
    help="The number of bins to use for the binned input style. Default is 51.",
)
parser.add_argument(
    "--max-seq-len",
    type=int,
    default=1536,
    help="The maximum length of the sequence. Default is 1000. The actual used "
    "max length would be the minimum of this value and the length of the longest "
    "sequence in the data.",
)
# omit the args for MLM and MVC, will always use them by default
parser.add_argument(
    "--training-tasks",  #  choices of "mlm", "gen", "both"
    type=str,
    default="both",
    choices=["pcpt", "gen", "both"],
    help="The tasks to use for training. pcpt: perception training with maked token "
    "learning. gen: generation. Default is both.",
)
parser.add_argument(
    "--mask-ratio",
    type=float,
    default=0.40,
    help="The ratio of masked values in the training data. Default is 0.40. This"
    "value will be ignored if --training-tasks is set to gen or both.",
)
parser.add_argument(
    "--trunc-by-sample",
    action="store_true",
    help="Whether to truncate the input by sampling rather than cutting off if "
    "sequence length > max_seq_length. Default is False.",
)
parser.add_argument(
    "--vocab-path",
    type=str,
    help="Path to the vocabulary file.",
)
# settings for training
parser.add_argument(
    "--local-rank",
    type=int,
    default=-1,
    help="The local rank of the process for using the torch.distributed.launch "
    "utility. Will be -1 if not running in distributed model.",
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=32,
    help="The batch size for training. Default is 32.",
)
parser.add_argument(
    "--eval-batch-size",
    type=int,
    default=32,
    help="The batch size for evaluation. Default is 32.",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=10,
    help="The number of epochs for training.",
)
parser.add_argument(
    "--lr",
    type=float,
    default=1e-3,
    help="The learning rate for training. Default is 1e-3.",
)
parser.add_argument(
    "--scheduler-interval",
    type=int,
    default=100,
    help="The interval iterations for updating the learning rate. Default is 100. "
    "This will only be used when warmup-ratio is 0.",
)
parser.add_argument(
    "--scheduler-factor",
    type=float,
    default=0.99,
    help="The factor for updating the learning rate. Default is 0.99. "
    "This will only be used when warmup-ratio is 0.",
)
parser.add_argument(
    "--warmup-ratio-or-step",
    type=float,
    default=0.1,
    help="The ratio of warmup steps out of the total training steps. Default is 0.1. "
    "If warmup-ratio is above 0, will use a cosine scheduler with warmup. If "
    "the value is above 1, will use it as the number of warmup steps.",
)
parser.add_argument(
    "--no-cls",
    action="store_true",
    help="Whether to deactivate the classification loss. Default is False.",
)
parser.add_argument(
    "--no-cce",
    action="store_true",
    help="Whether to deactivate the contrastive cell embedding objective. "
    "Default is False.",
)
parser.add_argument(
    "--fp16",
    action="store_true",
    help="Whether to train in automatic mixed precision. Default is False.",
)
parser.add_argument(
    "--fast-transformer",
    type=bool,
    default=True,
    help="Whether to use the fast transformer. Default is True.",
)

# settings for model
parser.add_argument(
    "--nlayers",
    type=int,
    default=4,
    help="The number of layers for the transformer. Default is 4.",
)
parser.add_argument(
    "--nheads",
    type=int,
    default=4,
    help="The number of heads for the transformer. Default is 4.",
)
parser.add_argument(
    "--embsize",
    type=int,
    default=64,
    help="The embedding size for the transformer. Default is 64.",
)
parser.add_argument(
    "--d-hid",
    type=int,
    default=64,
    help="dimension of the feedforward network model in the transformer. "
    "Default is 64.",
)
parser.add_argument(
    "--dropout",
    type=float,
    default=0.2,
    help="The dropout rate. Default is 0.2.",
)
parser.add_argument(
    "--n-layers-cls",
    type=int,
    default=3,
    help="The number of layers for the classification network, including the "
    "output layer. Default is 3.",
)

# settings for logging
parser.add_argument(
    "--log-interval",
    type=int,
    default=100,
    help="The interval for logging. Default is 100.",
)
parser.add_argument(
    "--save-interval",
    type=int,
    default=1000,
    help="The interval for saving the model. Default is 1000.",
)


if scg.utils.isnotebook():
    args = parser.parse_args(
        args=[
            "-d",
            "/scratch/hdd001/home/haotian/datasets/cellxgene/3faad104-2ab8-4434-816d-474d8d2641db.scb",
            "-s",
            "./save/tmp",
            "--batch-size",
            "16",
            "--max-seq-len",
            "512",
            "--trunc-by-sample",
            "--no-cls",
            "--no-cce",
        ]
    )
else:
    args = parser.parse_args()


# args.local_rank = os.environ['LOCAL_RANK']
# validate settings
assert args.input_style in ["normed_raw", "log1p", "binned"]
assert args.input_emb_style in ["category", "continuous", "scaling"]
if args.input_style == "binned":
    if args.input_emb_style == "scaling":
        raise ValueError("input_emb_style `scaling` is not supported for binned input.")
elif args.input_style == "log1p" or args.input_style == "normed_raw":
    if args.input_emb_style == "category":
        raise ValueError(
            "input_emb_style `category` is not supported for log1p or normed_raw input."
        )

if args.input_emb_style == "category":
    args.mask_value = args.n_bins + 1
    args.pad_value = args.n_bins  # for padding gene expr values
    n_input_bins = args.n_bins + 2
else:
    args.mask_value = -1
    args.pad_value = -2
    n_input_bins = args.n_bins

if args.training_tasks in ["gen", "both"]:
    args.mask_ratio = [0.25, 0.50, 0.75]

# %% settings
print(args)

special_tokens = [args.pad_token, "<cls>", "<eoc>"]
USE_CLS = not args.no_cls
USE_CCE = not args.no_cce
MVC = True
USE_GENERATIVE_TRAINING = True if args.training_tasks in ["gen", "both"] else False

IS_DATA_PARALLEL = args.local_rank != -1
if IS_DATA_PARALLEL:
    # These two lines is to solve issue #1 based on the suggestion from
    # https://discuss.pytorch.org/t/94382
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.local_rank)

    torch.distributed.init_process_group(
        backend="nccl",
        rank=args.local_rank,
        timeout=timedelta(hours=10),
    )
    # specify device 0 since the CUDA_VISIBLE_DEVICES is set to one GPU
    # https://discuss.pytorch.org/t/67488/4
    device = torch.device("cuda:0")
    n_gpu = torch.cuda.device_count()
    world_size = torch.distributed.get_world_size()
    logger.info(
        f"device: {device} in world size {world_size}, "
        f"visible gpu(s): {os.environ['CUDA_VISIBLE_DEVICES']}/{n_gpu}"
    )
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

save_dir = Path(args.save_dir)
if args.local_rank in [0, -1]:
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    # copy all uncommitted changes to the save dir
    os.system(
        f"git diff > {str(save_dir / 'git_diff_')}{scg.utils.get_git_commit()}.diff"
    )
if IS_DATA_PARALLEL:
    torch.distributed.barrier()

scg.utils.add_file_handler(logger, save_dir / "run.log")
# log running date and current git commit
logger.info(f"Running on {time.strftime('%Y-%m-%d %H:%M:%S')}")
logger.info(f"Current git commit: {scg.utils.get_git_commit()}")

writer = SummaryWriter(log_dir=save_dir / "tensorboard")
if IS_DATA_PARALLEL:
    writer = MainProcessOnly(writer)


# %% [markdown]
# # Load and prepare data


# TODO: move this to the preprocessing in DataBank
def _map_append_cls(dataset: Dataset) -> Dataset:
    logger.info(f"Rank {args.local_rank}: Appending <cls> to dataset")
    dataset = dataset.map(
        lambda example: {
            "genes": [vocab["<cls>"]] + example["genes"],
            "expressions": [args.pad_value] + example["expressions"],
        },
        # batched=True,  # not using since then the map func needs to loop
        num_proc=len(os.sched_getaffinity(0)),
    )

    return dataset


# Load data
if args.data_source.endswith("human"):
    TISSUE_LIST = [
        "heart",
        "blood",
        "brain",
        "lung",
        "kidney",
        "intestine",
        "pancreas",
        "others",
    ]
    root_data_source = Path(args.data_source).parent
    raw_dataset_list = []
    vocab = GeneVocab.from_file(Path(args.vocab_path))
    for tissue in TISSUE_LIST:
        tissue_data_path = root_data_source / tissue
        cls_prefix_datatable = (
            tissue_data_path / "all_counts" / "cls_prefix_data.parquet"
        )
        cache_dir = tissue_data_path / "cache"
        tissue_dataset = load_dataset(
            "parquet",
            data_files=str(cls_prefix_datatable),
            split="train",
            cache_dir=str(cache_dir),
        )
        logger.info(f"Loaded {tissue} examples from {cls_prefix_datatable}")
        raw_dataset_list.append(tissue_dataset)
    print("merging dataset...")
    raw_dataset = concatenate_datasets(raw_dataset_list)
    print("done merging dataset")
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

elif Path(args.data_source).is_dir() and args.data_source.endswith(".scb"):
    # the large-scale data structure
    db = DataBank.from_path(args.data_source)
    raw_dataset = db.main_data.data
    vocab: GeneVocab = db.gene_vocab
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)
    if USE_CCE or USE_CLS or MVC:
        # load or make the dataset w/ <cls> appended at the beginning
        cls_prefix_datatable = Path(args.data_source) / "cls_prefix_data.parquet"
        if not cls_prefix_datatable.exists():
            if args.local_rank in [0, -1]:
                raw_dataset = _map_append_cls(raw_dataset)
                raw_dataset.to_parquet(cls_prefix_datatable)
            if IS_DATA_PARALLEL:
                torch.distributed.barrier()  # wait for the mapping to finish
        raw_dataset = load_dataset(
            "parquet",
            data_files=str(cls_prefix_datatable),
            split="train",
            cache_dir=args.data_source,
        )
        logger.info(f"Loaded {len(raw_dataset)} examples from {cls_prefix_datatable}")
elif Path(args.data_source).is_dir():
    # collection of parquet files
    parquet_files = [str(f) for f in Path(args.data_source).glob("*.parquet")]
    cache_dir = Path(args.data_source).parent / "cache"
    vocab = GeneVocab.from_file(Path(args.vocab_path))
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)
    if USE_CCE or USE_CLS or MVC:
        # load or make the dataset w/ <cls> appended at the beginning
        cls_prefix_datatable = Path(args.data_source) / "cls_prefix_data.parquet"
        if not cls_prefix_datatable.exists():
            if args.local_rank in [0, -1]:
                logger.info(f"Rank {args.local_rank}: Preparing dataset")
                raw_dataset = load_dataset(
                    "parquet",
                    data_files=parquet_files,
                    split="train",
                    cache_dir=str(cache_dir),
                )
                raw_dataset = _map_append_cls(raw_dataset)
                raw_dataset.to_parquet(str(cls_prefix_datatable))
            if IS_DATA_PARALLEL:
                torch.distributed.barrier()  # wait for the mapping to finish
        raw_dataset = load_dataset(
            "parquet",
            data_files=str(cls_prefix_datatable),
            split="train",
            cache_dir=str(cache_dir),
        )
        logger.info(f"Loaded {len(raw_dataset)} examples from {cls_prefix_datatable}")
elif Path(args.data_source).is_file():
    adata = sc.read(args.data_source, cache=True)
    # Specific the required column names, when loading the data the first time.
    # Store the column names for later use.
    (
        celltype_col,
        str_celltype_col,
        gene_col,
        batch_key,
    ) = scg.utils.find_required_colums(
        adata,
        id=args.data_source,
        configs_dir=Path(args.data_source).parent,
    )
    if celltype_col is None:
        celltype_col = "int" + str_celltype_col
        adata.obs[celltype_col] = scg.utils.category_str2int(
            adata.obs[str_celltype_col]
        )
elif args.data_source == "test":  # Using test data
    raw_dataset = Dataset.from_dict(
        {
            "id": [1] * 300,
            "genes": [[1, 2, 3]] * 300,
            "expressions": [[1.0, 2.0, 3.0]] * 300,
        }
    )
    vocab = GeneVocab.from_dict({"zero": 0, "a": 1, "b": 2, "c": 3})
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

if args.load_model is not None:
    model_dir = Path(args.load_model)
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    if len(vocab) != len(json.load(open(model_dir / "vocab.json"))):
        raise ValueError(
            f"The vocabulary in the model directory to load ({model_dir}) does "
            "not match the current vocabulary. "
        )
    with open(model_config_file, "r") as f:
        model_configs = json.load(f)
    if args.pad_token != model_configs["pad_token"]:
        logger.warning(
            f"The pad token in the model directory to load ({model_dir}) "
            "does not match the current pad token. Be careful if this is not expected."
        )
    if args.pad_value != model_configs["pad_value"]:
        logger.warning(
            f"The pad value in the model directory to load ({model_dir}) "
            "does not match the current pad value. Be careful if this is not expected."
        )
    logger.info(
        f"Resume model from {model_file}, the model args will be overridden the "
        f"config {model_config_file}."
    )
    args.embsize = model_configs["embsize"]
    args.nheads = model_configs["nheads"]
    args.d_hid = model_configs["d_hid"]
    args.nlayers = model_configs["nlayers"]
    args.n_layers_cls = model_configs["n_layers_cls"]

    # resave the args with the new values
    if args.local_rank in [0, -1]:
        with open(save_dir / "args.json", "w") as f:
            json.dump(vars(args), f, indent=2)

# save the vocabulary
if args.local_rank in [0, -1]:
    with open(save_dir / "vocab.json", "w") as f:
        json.dump(
            {token: index for token, index in vocab.get_stoi().items()},
            f,
            indent=2,
        )
if IS_DATA_PARALLEL:
    torch.distributed.barrier()  # wait for saving all the files

# %% [markdown]
# # Data processing
# convert format to return torch.tensor
raw_dataset = raw_dataset.with_format("torch")

# split train and validation,
raw_dataset = raw_dataset.train_test_split(
    test_size=args.valid_size_or_ratio, shuffle=True
)
train_dataset = raw_dataset["train"]
valid_dataset = raw_dataset["test"]
logger.info(f"train set number of samples: {len(train_dataset)}, ")
logger.info(f"valid set number of samples: {len(valid_dataset)}, ")

# %% data loading
# data collator for online padding and sampling
# make separate two types of input and output
collator = scg.DataCollator(
    do_padding=True if args.max_seq_len is not None else False,
    pad_token_id=vocab[args.pad_token],
    pad_value=args.pad_value,
    do_mlm=True,
    do_binning=True if args.input_style == "binned" else False,
    mlm_probability=args.mask_ratio,
    mask_value=args.mask_value,
    max_length=args.max_seq_len,
    sampling=args.trunc_by_sample,
    data_style=args.training_tasks,
)

# TODO: try batch sampler, train_sampler = BatchSampler()
train_sampler = (
    DistributedSampler(train_dataset)
    if IS_DATA_PARALLEL
    else RandomSampler(train_dataset)
)
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    sampler=train_sampler,
    collate_fn=collator,
    drop_last=False,
    num_workers=min(len(os.sched_getaffinity(0)), args.batch_size),
    pin_memory=True,
    prefetch_factor=4,
)
valid_sampler = (
    DistributedSampler(valid_dataset, shuffle=False)
    if IS_DATA_PARALLEL
    else SequentialSampler(valid_dataset)
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=args.eval_batch_size,
    sampler=valid_sampler,
    collate_fn=collator,
    drop_last=False,
    num_workers=min(len(os.sched_getaffinity(0)), args.eval_batch_size),
    pin_memory=True,
)


# %% [markdown]
"""
## Notes
1. TODO: remember the distributed setting
https://huggingface.co/docs/datasets/v2.3.2/en/process#distributed-usage
2. [Dataset.format](https://huggingface.co/docs/datasets/v2.3.2/en/process#format) 
as pytorch conviniently convert to torch.tensors.

    ```python
    >>> dataset.reset_format()
    >>> dataset.format
    {'type': None,
    'format_kwargs': {},
    'columns': ['id', 'genes', 'expressions'],
    'output_all_columns': False}
    >>> dataset = dataset.with_format(type="pytorch")
    >>> dataset.format
    {'type': 'torch',
    'format_kwargs': {},
    'columns': ['id', 'genes', 'expressions'],
    'output_all_columns': False}
    >>> dataset[0]
    {'id': tensor(0),
    'genes': tensor([34797, 16936,  2745,  ..., 17076, 17078, 17072]),
    'expressions': tensor([1., 1., 1.,  ..., 8., 5., 7.])}
    ```
3. Instruction for using with pytorch and achieving the best performance, 
[here](https://huggingface.co/docs/datasets/v2.3.2/en/use_with_pytorch).
Some key points: 

    - Format to device cpu or gpu  
    - Use multiple loading processes
    - Use a BatchSampler
    - Personal suggestion: use the format_transform on the fly
"""

# %%
if USE_CLS:
    celltypes_labels = raw_dataset["celltypes"]
    num_types = len(set(celltypes_labels))
    celltypes_labels = np.array(celltypes_labels)

# # TODO: check gene and other statistics
# max_num_of_non_zero_genes = db.num_genes

# if args.local_rank in [0, -1]:
#     scg.utils.histogram(
#         torch.cat(train_dataset[:10000]["expressions"]).numpy(),
#         torch.cat(valid_dataset[:10000]["expressions"]).numpy(),
#         title="Histogram of clipped values",
#         save=save_dir / "histogram_clipped_values.png",
#     )
# if IS_DATA_PARALLEL:
#     torch.distributed.barrier()

# %% [markdown]
# # Create and train scGPT
ntokens = len(vocab)  # size of vocabulary
model = TransformerModel(
    ntokens,
    d_model=args.embsize,
    nhead=args.nheads,
    d_hid=args.d_hid,
    nlayers=args.nlayers,
    nlayers_cls=args.n_layers_cls,
    n_cls=num_types if USE_CLS else 1,
    vocab=vocab,
    dropout=args.dropout,
    pad_token=args.pad_token,
    pad_value=args.pad_value,
    do_mvc=MVC,
    do_dab=False,
    use_batch_labels=False,  # TODO: try using batch labels, may help MVC
    input_emb_style=args.input_emb_style,
    n_input_bins=n_input_bins,
    use_generative_training=USE_GENERATIVE_TRAINING,
    use_fast_transformer=args.fast_transformer,
    fast_transformer_backend="flash",
)
if args.load_model is not None:
    try:
        model.load_state_dict(torch.load(model_file))
    except:
        from collections import OrderedDict

        params = OrderedDict()
        for key, value in torch.load(model_file).items():
            params[key.replace("module.", "")] = value
        model.load_state_dict(params)
model.to(device)
logger.info(model)
if IS_DATA_PARALLEL:
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[device],
        output_device=device,
        find_unused_parameters=False,
    )


criterion = masked_mse_loss
criterion_cls = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# setup scheduler
if args.warmup_ratio_or_step > 0:
    total_num_batches = len(train_loader) * args.epochs
    warmup_steps = (
        int(total_num_batches * args.warmup_ratio_or_step)
        if args.warmup_ratio_or_step < 1
        else int(args.warmup_ratio_or_step)
    )
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_num_batches,
        last_epoch=-1,
    )
else:
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, args.scheduler_interval, gamma=args.scheduler_factor
    )

# amp fp16 training
scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)


def train(model: nn.Module, train_loader: DataLoader, epoch: int) -> None:
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss, total_mse, total_cls, total_gen, total_mvc = 0.0, 0.0, 0.0, 0.0, 0.0
    total_error = 0.0
    log_interval = args.log_interval
    start_time = time.time()

    num_batches = len(train_loader)
    for batch, data_dict in enumerate(train_loader):
        global_iter = epoch * num_batches + batch

        data_dict = {k: v.to(device) for k, v in data_dict.items()}
        if USE_GENERATIVE_TRAINING:
            pcpt_gene = data_dict["pcpt_gene"]
            pcpt_expr = data_dict["pcpt_expr"]
            pcpt_key_padding_mask = pcpt_gene.eq(vocab[args.pad_token])
            gen_gene = data_dict["gen_gene"]
            gen_expr_target = target_values = data_dict["gen_expr_target"]
            gen_key_padding_mask = gen_gene.eq(vocab[args.pad_token])
        else:
            input_gene_ids = data_dict["gene"]
            input_values = data_dict["masked_expr"]
            target_values = data_dict["expr"]
            src_key_padding_mask = input_gene_ids.eq(vocab[args.pad_token])

        with torch.cuda.amp.autocast(enabled=args.fp16):
            if USE_GENERATIVE_TRAINING:
                output_dict = model(
                    pcpt_gene,
                    pcpt_expr,
                    pcpt_key_padding_mask,
                    gen_gene,
                    gen_key_padding_mask,
                    CLS=USE_CLS,
                    MVC=MVC,
                    generative_training=True,
                )
                gen_expr_preds = output_values = output_dict["gen_preds"]

                positions_to_match = ~gen_key_padding_mask
                loss = loss_mse = criterion(
                    gen_expr_preds, gen_expr_target, positions_to_match
                )
                writer.add_scalar("train/mse", loss_mse, global_iter)
                if MVC:
                    loss_mvc = criterion(
                        output_dict["mvc_output"][:, pcpt_gene.shape[1] :],
                        gen_expr_target,
                        positions_to_match,
                    )
                    loss = loss + loss_mvc
                    writer.add_scalar("train/mvc", loss_mvc, global_iter)
            else:
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    CLS=USE_CLS,
                    CCE=USE_CCE,  # TODO: move these flags to model's attributes
                    MVC=MVC,
                    generative_training=False,
                )
                output_values = output_dict["mlm_output"]

                positions_to_match = input_values.eq(
                    args.mask_value
                )  # the postions to predict
                loss = loss_mse = criterion(
                    output_values, target_values, positions_to_match
                )
                writer.add_scalar("train/mse", loss_mse, global_iter)
                if USE_CLS:
                    target_labels = data_dict["celltypes"]
                    loss_cls = criterion_cls(output_dict["cls_output"], target_labels)
                    loss = loss + loss_cls
                    writer.add_scalar("train/cls", loss_cls, global_iter)
                if USE_CCE:
                    loss_cce = 10 * output_dict["loss_cce"]
                    loss = loss + loss_cce
                    writer.add_scalar("train/cce", loss_cce, global_iter)
                if MVC:
                    loss_mvc = criterion(
                        output_dict["mvc_output"], target_values, positions_to_match
                    )
                    loss = loss + loss_mvc
                    writer.add_scalar("train/mvc", loss_mvc, global_iter)
            writer.add_scalar("train/loss", loss, global_iter)

            if USE_GENERATIVE_TRAINING and global_iter > 1000:
                previous_cell_embs = output_dict["cell_emb"].detach()
                preds = model(
                    pcpt_gene,
                    pcpt_expr,
                    pcpt_key_padding_mask,
                    gen_gene,
                    gen_key_padding_mask,
                    CLS=False,
                    MVC=False,
                    input_cell_emb=previous_cell_embs,
                    generative_training=True,
                )["gen_preds"]
                loss_gen = criterion(preds, gen_expr_target, positions_to_match)
                loss = loss + loss_gen
                writer.add_scalar("train/gen", loss_gen, global_iter)

                # TODO: try this choice of using a separate backprop
                # # this part is for the choice of using a separate backprop
                # model.zero_grad()
                # scaler.scale(loss_gen).backward()
                # scaler.unscale_(optimizer)
                # torch.nn.utils.clip_grad_norm_(
                #     model.parameters(),
                #     1.0,
                #     error_if_nonfinite=False if scaler.is_enabled() else True,
                # )
                # scaler.step(optimizer)
                # scaler.update()

        if args.grad_accu_steps > 1:
            loss = loss / args.grad_accu_steps
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if args.grad_accu_steps > 1:
            if batch % args.grad_accu_steps == 0 or batch == num_batches - 1:
                scheduler.step()
                optimizer.zero_grad()
        else:
            scheduler.step()
            optimizer.zero_grad()

        with torch.no_grad():
            mre = masked_relative_error(
                output_values, target_values, positions_to_match
            )
            writer.add_scalar("train/mre", mre, global_iter)

        total_loss += loss.item()
        total_mse += loss_mse.item()
        total_cls += loss_cls.item() if USE_CLS else 0.0
        total_gen += loss_gen.item() if "loss_gen" in locals() else 0.0
        total_mvc += loss_mvc.item() if MVC else 0.0
        total_error += mre.item()
        if args.local_rank in [0, -1] and batch % log_interval == 0 and batch > 0:
            # Writer logs gradients distribution
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    writer.add_histogram(name + "_grad", param.grad, global_iter)
                    writer.add_histogram(name + "_param", param, global_iter)

            # Log scalar values
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_mse = total_mse / log_interval
            cur_cls = total_cls / log_interval if USE_CLS else 0.0
            cur_gen = total_gen / log_interval if "loss_gen" in locals() else 0.0
            cur_mvc = total_mvc / log_interval if MVC else 0.0
            cur_error = total_error / log_interval
            # ppl = math.exp(cur_loss)
            logger.info(
                f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.2f} | mse {cur_mse:5.2f} | mre {cur_error:5.2f} |"
                + (f"cls {cur_cls:5.2f} | " if USE_CLS else "")
                + (f"gen {cur_gen:5.2f} |" if "loss_gen" in locals() else "")
                + (f"mvc {cur_mvc:5.2f} |" if MVC else "")
            )
            writer.add_scalar("lr", lr, global_iter)

            total_loss = 0
            total_mse = 0
            total_cls = 0
            total_gen = 0
            total_mvc = 0
            total_error = 0
            start_time = time.time()

        # immediately eval and save
        if batch % args.save_interval == 0 and batch > 0:
            eval_and_save(model, valid_loader, global_iter)
            model.train()  # important, reset to train mode


def evaluate(model: nn.Module, valid_loader: DataLoader) -> Dict[str, torch.Tensor]:
    """
    Evaluate the model on the evaluation data.
    """
    model.eval()
    total_loss = 0.0
    total_error = 0.0
    with torch.no_grad():
        for data_dict in valid_loader:
            data_dict = {k: v.to(device) for k, v in data_dict.items()}
            if USE_GENERATIVE_TRAINING:
                pcpt_gene = data_dict["pcpt_gene"]
                pcpt_expr = data_dict["pcpt_expr"]
                pcpt_key_padding_mask = pcpt_gene.eq(vocab[args.pad_token])
                gen_gene = data_dict["gen_gene"]
                gen_expr_target = target_values = data_dict["gen_expr_target"]
                gen_key_padding_mask = gen_gene.eq(vocab[args.pad_token])
            else:
                input_gene_ids = data_dict["gene"]
                input_values = data_dict["masked_expr"]
                target_values = data_dict["expr"]
                src_key_padding_mask = input_gene_ids.eq(vocab[args.pad_token])

            with torch.cuda.amp.autocast(enabled=args.fp16):
                if USE_GENERATIVE_TRAINING:
                    output_dict = model(
                        pcpt_gene,
                        pcpt_expr,
                        pcpt_key_padding_mask,
                        gen_gene,
                        gen_key_padding_mask,
                        CLS=False,
                        MVC=False,
                        generative_training=True,
                    )
                    gen_expr_preds = output_values = output_dict["gen_preds"]

                    positions_to_match = ~gen_key_padding_mask
                else:
                    output_dict = model(
                        input_gene_ids,
                        input_values,
                        src_key_padding_mask=src_key_padding_mask,
                        CLS=False,  # evaluation does not need CLS or CCE
                        CCE=False,
                        MVC=False,
                        generative_training=False,
                    )
                    output_values = output_dict["mlm_output"]
                    positions_to_match = input_values.eq(args.mask_value)

                loss = criterion(output_values, target_values, positions_to_match)
            total_loss += loss.item()
            total_error += masked_relative_error(
                output_values, target_values, positions_to_match
            ).item()
    total_loss = total_loss / len(valid_loader)
    total_error = total_error / len(valid_loader)
    return {
        "mse": torch.tensor(total_loss, device=device, dtype=torch.float),
        "mre": torch.tensor(total_error, device=device, dtype=torch.float),
    }


def eval_and_save(
    model: nn.Module,
    valid_loader: DataLoader,
    iter_or_epoch: int,
    is_epoch: bool = False,
    save: bool = True,
) -> None:
    # perform evaluation in distributed data parallel
    val_loss, val_mre = evaluate(model, valid_loader).values()
    if IS_DATA_PARALLEL:
        # gather the results from all the processes
        val_loss_list = [torch.zeros_like(val_loss) for _ in range(world_size)]
        val_mre_list = [torch.zeros_like(val_mre) for _ in range(world_size)]
        torch.distributed.all_gather(val_loss_list, val_loss)
        torch.distributed.all_gather(val_mre_list, val_mre)
        val_loss = torch.mean(torch.stack(val_loss_list))
        val_mre = torch.mean(torch.stack(val_mre_list))
    val_loss, val_mre = val_loss.item(), val_mre.item()
    if args.local_rank in [0, -1]:
        if is_epoch:
            elapsed = time.time() - epoch_start_time
            logger.info("-" * 89)
            logger.info(
                f"| end of epoch {iter_or_epoch:3d} | time: {elapsed:5.2f}s | "
                f"valid loss/mse {val_loss:5.4f} | mre {val_mre:5.4f}"
            )
            logger.info(f"{'-' * 89}\n")
            writer.add_scalar("valid/mse", val_loss, iter_or_epoch * len(valid_loader))
            writer.add_scalar("valid/mre", val_mre, iter_or_epoch * len(valid_loader))
        else:
            logger.info(f"valid loss/mse {val_loss:5.4f} | mre {val_mre:5.4f}")
            writer.add_scalar("valid/mse", val_loss, iter_or_epoch)
            writer.add_scalar("valid/mre", val_mre, iter_or_epoch)

        global best_val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # save the best model
            logger.info(f"Saving the best model to {args.save_dir}")
            torch.save(
                model.module.state_dict()
                if isinstance(
                    model, (nn.DataParallel, nn.parallel.DistributedDataParallel)
                )
                else model.state_dict(),
                args.save_dir + "/best_model.pt",
            )

        if save:
            torch.save(
                model.module.state_dict()
                if isinstance(
                    model, (nn.DataParallel, nn.parallel.DistributedDataParallel)
                )
                else model.state_dict(),
                args.save_dir + f"/model-{'ep' if is_epoch else ''}{iter_or_epoch}.pt",
            )
    if IS_DATA_PARALLEL:
        torch.distributed.barrier()


# %%
best_val_loss = float("inf")
logger.info("Start training")
for epoch in range(1, args.epochs + 1):
    epoch_start_time = time.time()
    train(model, train_loader, epoch=epoch)
    eval_and_save(model, valid_loader, iter_or_epoch=epoch, is_epoch=True)

writer.flush()
writer.close()

# %%
# compare with the naive baseline of all ones
data_dict = next(iter(valid_loader))
input_values = data_dict["masked_expr"]
tagert_values = data_dict["expr"]
predict_ones = torch.ones(input_values.shape, dtype=torch.float32)
mse = masked_mse_loss(predict_ones, tagert_values, input_values.eq(args.mask_value))
mre = masked_relative_error(
    predict_ones, tagert_values, input_values.eq(args.mask_value)
)
logger.info(f"MSE: {mse.item()}, MRE: {mre.item()}")

# %% [markdown]
# # Analysis
model.to(device)
model.eval()

# %% [markdown]
# ## Cell embeddings


# def map_transform(examples: Dict[str, List]) -> Dict[str, torch.Tensor]:
#     """
#     Transform batch examples to a tensor.
#     """
#     # tensorize the examples
#     examples = {
#         k: [torch.tensor(v_i) for v_i in v] for k, v in examples.items() if k != "id"
#     }
#     examples = collator(examples)
#     return examples


# valid_dataset.set_transform(map_transform)

# cell_gene_embeddings = model.encoder(valid_dataset[:10000]["gene"].to(device))
# cell_gene_embeddings = cell_gene_embeddings.detach().cpu().numpy()

# cell_embeddings = np.mean(cell_gene_embeddings, axis=1)
