import os
from pathlib import Path
import sys

import torch


sys.path.insert(0, "../")
#

from datasets import Dataset, load_dataset

from transformers import TrainingArguments

# from scgpt.huggingface_model import scGPT_config, scGPT_ForPretraining
from scgpt.model.huggingface_model import scGPT_config, scGPT_ForPretraining
from scgpt.huggingface_data_collator import scGPT_DataCollator
from scgpt.huggingface_trainer import scGPT_pretrainingTrainer
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


# raw_dataset = Dataset.from_dict(
#     {
#         "id": [1] * 3000,
#         "genes": [[1, 2, 3]] * 3000,
#         "expressions": [[1.0, 2.0, 3.0]] * 3000,
#     }
# )

data_source = Path("/home/pangkuan/dev/data_disk/scb_sample/partition_0.scb")
db = DataBank.from_path(data_source)
raw_dataset = db.main_data.data
vocab: GeneVocab = db.gene_vocab
for s in special_tokens:
    if s not in vocab:
        vocab.append_token(s)
    # load or make the dataset w/ <cls> appended at the beginning
cls_prefix_datatable = data_source / "cls_prefix_data.parquet"


config = scGPT_config(vocab_size=len(vocab),
                      padding_idx=vocab["<pad>"],
                      )
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


# raw_dataset = Dataset.from_dict(
#     {
#         "id": [1] * 3000,
#         "genes": [[i for i in range(1200)]] * 3000,
#         "expressions": [[float(i) for i in range(1200)]] * 3000,
#     }
# )
raw_dataset = raw_dataset.with_format("torch")
raw_dataset = raw_dataset.train_test_split(test_size=0.2, shuffle=True)
train_dataset = raw_dataset["train"]
valid_dataset = raw_dataset["test"]
# vocab = GeneVocab.from_dict({"zero": 0, "a": 1, "b": 2, "c": 3})
# for s in special_tokens:
#     if s not in vocab:
#         vocab.append_token(s)


collator = scGPT_DataCollator(
    vocab,
    pad_token_id = vocab["<pad>"],
    mask_value=model.config.mask_value,
    max_length=1200,
    data_style="both",
    mlm_probability=0.15,
)
# train_loader = DataLoader(
#     train_dataset,
#     batch_size=16,
#     # sampler=train_sampler,
#     collate_fn=collator,
#     drop_last=False,
#     pin_memory=True,
#     # prefetch_factor=4,
# )

model = model.cuda()
# with torch.cuda.amp.autocast(enabled=True):
#     for data_dict in train_loader:
#         print(data_dict)
#         pcpt_gene = data_dict["pcpt_gene"].to(model.device)
#         pcpt_expr = data_dict["pcpt_expr"].to(model.device)
#         pcpt_key_padding_mask = pcpt_gene.eq(model.config.pad_value).to(model.device)
#         gen_gene = data_dict["gen_gene"].to(model.device)
#         gen_expr_target = target_values = data_dict["gen_expr_target"].to(model.device)
#         gen_key_padding_mask = gen_gene.eq(model.config.pad_value).to(model.device)

#         outputs = model(
#             pcpt_gene,
#             pcpt_expr,
#             pcpt_key_padding_mask,
#             gen_gene,
#             gen_key_padding_mask,
#             generative_training=True,
#         )


# for data in train_loader:
#     # print(data)
#     pcpt_gene = data["pcpt_gene"]
#     # if len(pcpt_gene) == 0:
#     #     print("empty")
#     #     break
#     print(len(pcpt_gene))
#     # break

training_args = TrainingArguments(
    per_gpu_train_batch_size=32,
    num_train_epochs=1,
    output_dir="./save",
    logging_dir="./logs",
    # debug=True,
    remove_unused_columns=False,
    fp16=True,
)

trainer = scGPT_pretrainingTrainer(
    model=model,
    args=training_args,
    data_collator=collator,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
)

print("start training...")

trainer.train()
