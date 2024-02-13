import json
import os
from pathlib import Path
import sys

import torch


sys.path.insert(0, "../")
#

from datasets import Dataset, load_dataset, concatenate_datasets

from transformers import TrainingArguments

# from scgpt.huggingface_model import scGPT_config, scGPT_ForPretraining
from scgpt.model.huggingface_model import scGPT_config, scGPT_ForPretraining
from scgpt.huggingface_data_collator import scGPT_DataCollator
from scgpt.huggingface_trainer import scGPT_pretrainingTrainer, scGPT_TrainingArguments
from torch.utils.data import DataLoader, BatchSampler, RandomSampler, SequentialSampler
from scgpt.tokenizer import GeneVocab, random_mask_value
from scgpt.scbank.databank import DataBank


special_tokens = ["<pad>", "<cls>", "<eoc>"]


def _map_append_cls(dataset: Dataset) -> Dataset:
    dataset = dataset.map(
        lambda example: {
            "genes": [vocab["<cls>"]] + example["genes"],
            "expressions": [0] + example["expressions"],
        },
        # batched=True,  # not using since then the map func needs to loop
        num_proc=len(os.sched_getaffinity(0)),
    )

    return dataset


MODEL_CONFIG = ""
TRAINING_ARGS = ""
data_source = Path("")
vocab_path = ""



if data_source.endswith("human"):
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
    root_data_source = Path(data_source).parent
    raw_dataset_list = []
    vocab = GeneVocab.from_file(Path(vocab_path))
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
        raw_dataset_list.append(tissue_dataset)
    print("merging dataset...")
    raw_dataset = concatenate_datasets(raw_dataset_list)
    print("done merging dataset")

for s in special_tokens:
    if s not in vocab:
        vocab.append_token(s)
    # load or make the dataset w/ <cls> appended at the beginning
cls_prefix_datatable = data_source / "cls_prefix_data.parquet"

config = scGPT_config.from_json_file(MODEL_CONFIG)
model = scGPT_ForPretraining(config)
if not cls_prefix_datatable.exists():
    raw_dataset = _map_append_cls(raw_dataset)
    raw_dataset.to_parquet(cls_prefix_datatable)


raw_dataset = raw_dataset.with_format("torch")
raw_dataset = raw_dataset.train_test_split(test_size=0.2, shuffle=True)
train_dataset = raw_dataset["train"]
valid_dataset = raw_dataset["test"]

with open(TRAINING_ARGS) as fin:
    args_json = json.load(fin)
training_args = scGPT_TrainingArguments(**args_json)


collator = scGPT_DataCollator(
    vocab,
    pad_token_id=vocab["<pad>"],
    mask_value=model.config.mask_value,
    max_length=training_args.max_length,
    data_style="both",
    mlm_probability=training_args.mlm_probability,
)


model = model.cuda()


trainer = scGPT_pretrainingTrainer(
    model=model,
    args=training_args,
    data_collator=collator,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
)

print("start training...")

trainer.train()
