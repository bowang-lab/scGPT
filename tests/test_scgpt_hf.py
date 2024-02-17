import json
import os
from pathlib import Path
import sys

import torch


sys.path.insert(0, "../")
#

from datasets import Dataset, load_dataset

from transformers import get_cosine_schedule_with_warmup

# from scgpt.huggingface_model import scGPT_config, scGPT_ForPretraining
from scgpt.model.huggingface_model import scGPT_config, scGPT_ForPretraining
from scgpt.huggingface_data_collator import scGPT_DataCollator
from scgpt.huggingface_trainer import scGPT_pretrainingTrainer, scGPT_TrainingArguments
from scgpt.tokenizer import GeneVocab
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


MODEL_CONFIG = "/home/pangkuan/dev/scGPT-release/tests/test_configs/model_config.json"
TRAINING_ARGS = "/home/pangkuan/dev/scGPT-release/tests/test_configs/training_args.json"
data_source = Path("/home/pangkuan/dev/data_disk/scb_sample/partition_0.scb")

db = DataBank.from_path(data_source)
raw_dataset = db.main_data.data
vocab: GeneVocab = db.gene_vocab
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
raw_dataset = load_dataset(
    "parquet",
    data_files=str(cls_prefix_datatable),
    split="train",
    cache_dir=data_source,
)

raw_dataset = raw_dataset.with_format("torch")

raw_dataset = raw_dataset.train_test_split(test_size=0.2, shuffle=True)
train_dataset = raw_dataset["train"]
valid_dataset = raw_dataset["test"]

# keep 10% of the dataset for testing the code
train_dataset = train_dataset.select(range(int(len(train_dataset) * 0.01)))
valid_dataset = valid_dataset.select(range(int(len(valid_dataset) * 0.01)))

with open(TRAINING_ARGS) as fin:
    args_json = json.load(fin)
training_args = scGPT_TrainingArguments(**args_json)
# print(training_args)
optimizer = torch.optim.Adam(model.parameters(), lr=training_args.learning_rate)

# if training_args.warmup_ratio_or_step > 0:
# total_num_batches = len(train_loader) * args.epochs
total_num_batches = len(train_dataset) / training_args.per_device_train_batch_size
warmup_steps = (
    int(total_num_batches * training_args.warmup_ratio_or_step)
    if training_args.warmup_ratio_or_step < 1
    else int(training_args.warmup_ratio_or_step)
)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_num_batches,
    last_epoch=-1,
)
# else:
#    scheduler = torch.optim.lr_scheduler.StepLR(
#        optimizer, training_args.scheduler_interval, gamma=training_args.scheduler_factor
#    )

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
    optimizers=(optimizer, scheduler),
)
# print(dir(trainer))
# print(trainer.args)

print("start training...")

trainer.train()
