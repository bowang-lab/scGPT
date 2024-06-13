import json
from typing import Optional
from torch import nn
from transformers import Trainer, TrainingArguments
import torch
from .loss import masked_mse_loss
from dataclasses import dataclass


@dataclass
class scGPT_TrainingArguments(TrainingArguments):
    mlm_probability: Optional[float] = 0.50
    max_length: Optional[int] = 1200
    warmup_ratio_or_step: Optional[int] = 0.1
    MVC: Optional[bool] = False
    evaluation_strategy: Optional[str] = "steps"
    # TODO: add custom arguments here

    # training loss, mvc loss, etc.
    def from_json_file(self, json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
        for k, v in data.items():
            setattr(self, k, v)


def compute_metrics(eval_pred):
    # logits, labels = eval_pred
    raise NotImplementedError


class scGPT_pretrainingTrainer(Trainer):
    def compute_loss(self, model, data_dict, return_outputs=False):
        # print("compute_loss")
        # unpack data dict
        pcpt_gene = data_dict["pcpt_gene"].int()
        # pcpt_expr = data_dict["pcpt_expr"]

        gen_gene = data_dict["gen_gene"].int()
        gen_expr_target = data_dict["gen_expr_target"]
        gen_key_padding_mask = data_dict["gen_key_padding_mask"]
        # pcpt_key_padding_mask = data_dict["pcpt_key_padding_mask"]
        data_dict["CLS"] = False
        data_dict["MVC"] = self.args.MVC
        data_dict["generative_training"] = True

        # forward pass of generation
        outputs = model(**data_dict)

        gen_expr_preds = outputs.get("gen_preds")
        positions_to_match = ~gen_key_padding_mask
        loss = loss_mse = masked_mse_loss(
            gen_expr_preds, gen_expr_target, positions_to_match
        )

        loss_mvc = None
        if self.args.MVC:
            # MVC loss
            loss_mvc = masked_mse_loss(
                outputs.get("mvc_output")[:, pcpt_gene.shape[1] :],
                gen_expr_target,
                positions_to_match,
            )
            loss = loss + loss_mvc

        # embed -> expr loss
        previous_cell_embs = outputs.get("cell_emb").detach()
        data_dict["cell_emb"] = previous_cell_embs
        preds = self.model(**data_dict).get("gen_preds")
        loss_gen = masked_mse_loss(preds, gen_expr_target, positions_to_match)
        loss = loss + loss_gen

        outputs = {
            "loss": loss,
            "loss_mse": loss_mse,
            "loss_mvc": loss_mvc,
            "loss_gen": loss_gen,
            "gen_preds": gen_expr_preds,
            "cell_emb": previous_cell_embs,
        }

        return (loss, outputs) if return_outputs else loss
