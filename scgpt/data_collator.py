from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import torch
import numpy as np

from .preprocess import binning


@dataclass
class DataCollator:
    """
    Data collator for the mask value learning task. It pads the sequences to
    the maximum length in the batch and masks the gene expression values.

    Args:
        do_padding (:obj:`bool`): whether to pad the sequences to the max length.
        pad_token_id (:obj:`int`, optional): the token id to use for padding.
            This is required if do_padding is True.
        pad_value (:obj:`int`): the value to use for padding the expression
            values to the max length.
        do_mlm (:obj:`bool`): whether to do masking with MLM.
        do_binning (:obj:`bool`): whether to bin the expression values.
        mlm_probability (:obj:`float`): the probability of masking with MLM.
        mask_value (:obj:`int`): the value to fill at the expression postions
            that are masked.
        max_length (:obj:`int`, optional): the maximum length of the sequences.
            This is required if do_padding is True.
        sampling (:obj:`bool`): whether to do sampling instead of truncation if
            length > max_length.
        keep_first_n_tokens (:obj:`int`): the number of tokens in the beginning
            of the sequence to keep unchanged from sampling. This is useful when
            special tokens have been added to the beginning of the sequence.
            Default to 1.
    """

    do_padding: bool = True
    pad_token_id: Optional[int] = None
    pad_value: int = 0
    do_mlm: bool = True
    do_binning: bool = True
    mlm_probability: float = 0.15
    mask_value: int = -1
    max_length: Optional[int] = None
    sampling: bool = True
    keep_first_n_tokens: int = 1

    def __post_init__(self):
        if self.do_padding:
            if self.pad_token_id is None:
                raise ValueError("`pad_token_id` is required if `do_padding`.")
            if self.max_length is None:
                raise ValueError("`max_length` is required if `do_padding`.")

        if self.mlm_probability <= 0 or self.mlm_probability >= 1:
            raise ValueError("`mlm_probability` must be between 0 and 1.")

        if self.keep_first_n_tokens < 0 or self.keep_first_n_tokens > self.max_length:
            raise ValueError(
                "`keep_first_n_tokens` must be between 0 and `max_length` "
                f"({self.max_length})."
            )

    def __call__(
        self, examples: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Each example is like:
            {'id': tensor(184117),
            'genes': tensor([36572, 17868, ..., 17072]),
            'expressions': tensor([ 0.,  2., ..., 18.])}
        """
        if not isinstance(examples[0], Mapping):
            return NotImplementedError

        device = examples[0]["genes"].device

        max_ori_len = max(len(example["genes"]) for example in examples)
        _max_length = self.max_length if max_ori_len >= self.max_length else max_ori_len

        # pad and truncate
        padded_genes = []
        padded_expressions = []
        for i in range(len(examples)):
            genes = examples[i]["genes"]
            expressions = examples[i]["expressions"]
            if self.do_binning:
                expressions[self.keep_first_n_tokens :] = binning(
                    row=expressions[self.keep_first_n_tokens :],
                    n_bins=51,
                )
            genes, expressions = self._sample_or_truncate_plus_pad(
                genes, expressions, _max_length
            )  # torch tensors of length _max_length
            padded_genes.append(genes)
            padded_expressions.append(expressions)

        padded_genes = torch.stack(padded_genes, dim=0).to(device)
        padded_expressions = torch.stack(padded_expressions, dim=0).to(device)

        data_dict = {
            "gene": padded_genes,
            "expr": padded_expressions,
        }

        # mask
        if self.do_mlm:
            masked_expressions = self._mask(padded_expressions)
        else:
            masked_expressions = padded_expressions
        data_dict["masked_expr"] = masked_expressions

        return data_dict

    def _mask(self, expressions: torch.Tensor) -> torch.Tensor:
        """
        Mask the expression values with MLM.
        """
        device = expressions.device
        shape = expressions.shape

        probability_matrix = torch.full(shape, self.mlm_probability)
        # set padded postion probability to 0
        probability_matrix[expressions.eq(self.pad_value)] = 0
        if self.keep_first_n_tokens > 0:
            probability_matrix[:, : self.keep_first_n_tokens] = 0

        mask = torch.bernoulli(probability_matrix).bool()
        mask = mask.to(device)

        masked_expressions = expressions.masked_fill(mask, self.mask_value)
        return masked_expressions

    def _sample_or_truncate_plus_pad(
        self,
        genes: torch.LongTensor,
        expressions: torch.Tensor,
        max_length: int,
    ) -> Tuple[torch.LongTensor, torch.Tensor]:
        assert len(genes) == len(expressions)
        if len(genes) == max_length:
            return genes, expressions
        if len(genes) > max_length:  # sample or truncate
            if self.sampling:
                return self._sample(genes, expressions, max_length)
            else:
                return genes[:max_length], expressions[:max_length]
        else:  # pad
            return self._pad(genes, expressions, max_length)

    def _sample(
        self,
        genes: torch.LongTensor,
        expressions: torch.Tensor,
        max_length: int,
    ) -> Tuple[torch.LongTensor, torch.Tensor]:
        # NOTE: the fastest way to sample in torch has been benchmarked here
        # https://discuss.pytorch.org/t/torch-equivalent-of-numpy-random-choice/16146/19
        # it shows the randperm on gpu is the fastest.
        # NOTE: also, the current implementation permute the orders of the genes
        # and expressions, although it is probably a nice argmentation.
        device = genes.device
        if self.keep_first_n_tokens == 0:
            indices = torch.randperm(len(genes), device=device)[:max_length]
            return genes[indices], expressions[indices]

        # keep the first n tokens unchanged
        _n = self.keep_first_n_tokens
        indices = torch.randperm(len(genes) - _n, device=device)[: max_length - _n]
        indices = torch.cat([torch.arange(_n), indices + _n], dim=0)
        return genes[indices], expressions[indices]

    def _pad(
        self,
        genes: torch.LongTensor,
        expressions: torch.Tensor,
        max_length: int,
    ):
        device = genes.device
        genes = torch.cat(
            [
                genes,
                torch.full(
                    (max_length - len(genes),),
                    self.pad_token_id,
                    dtype=genes.dtype,
                    device=device,
                ),
            ]
        )
        expressions = torch.cat(
            [
                expressions,
                torch.full(
                    (max_length - len(expressions),),
                    self.pad_value,
                    dtype=expressions.dtype,
                    device=device,
                ),
            ]
        )
        return genes, expressions
