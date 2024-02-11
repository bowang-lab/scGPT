import sys

import torch


sys.path.insert(0, "../")
from scgpt.tokenizer import GeneVocab, random_mask_value
from datasets import Dataset

# from scgpt.huggingface_model import scGPT_config, scGPT_ForPretraining
from scgpt.model.huggingface_model import scGPT_config, scGPT_ForPretraining
from scgpt.data_collator import DataCollator
from torch.utils.data import DataLoader, BatchSampler, RandomSampler, SequentialSampler


config = scGPT_config()
model = scGPT_ForPretraining(config)

special_tokens = ["<pad>", "<cls>", "<eoc>"]

raw_dataset = Dataset.from_dict(
    {
        "id": [1] * 300,
        "genes": [[1, 2, 3]] * 300,
        "expressions": [[1.0, 2.0, 3.0]] * 300,
    }
)
raw_dataset = raw_dataset.with_format("torch")
raw_dataset = raw_dataset.train_test_split(test_size=0.2, shuffle=True)
train_dataset = raw_dataset["train"]
valid_dataset = raw_dataset["test"]
vocab = GeneVocab.from_dict({"zero": 0, "a": 1, "b": 2, "c": 3})
for s in special_tokens:
    if s not in vocab:
        vocab.append_token(s)


collator = DataCollator(
    vocab,
    model.config.pad_value,
    model.config.mask_value,
    max_length=1200,
    data_style="both",
    mlm_probability=0.5,
)
train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    # sampler=train_sampler,
    collate_fn=collator,
    drop_last=False,
    pin_memory=True,
    # prefetch_factor=4,
)

model = model.cuda()
with torch.cuda.amp.autocast(enabled=True):
    for data_dict in train_loader:
        print(data_dict)
        pcpt_gene = data_dict["pcpt_gene"].to(model.device)
        pcpt_expr = data_dict["pcpt_expr"].to(model.device)
        pcpt_key_padding_mask = pcpt_gene.eq(model.config.pad_value).to(model.device)
        gen_gene = data_dict["gen_gene"].to(model.device)
        gen_expr_target = target_values = data_dict["gen_expr_target"].to(model.device)
        gen_key_padding_mask = gen_gene.eq(model.config.pad_value).to(model.device)

        outputs = model(
            pcpt_gene,
            pcpt_expr,
            pcpt_key_padding_mask,
            gen_gene,
            gen_key_padding_mask,
            generative_training=True,
        )
