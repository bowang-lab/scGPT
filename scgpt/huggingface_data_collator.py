from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerBase

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import torch
import numpy as np
import sys

sys.path.insert(0, "../")
from scgpt.preprocess import binning

class scGPT_DataCollator(DataCollatorForLanguageModeling):
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
        reserve_keys (:obj:`List[str]`, optional): a list of keys in the examples
            to reserve in the output dictionary. Default to []. These fields
            will be kept unchanged in the output.
        keep_first_n_tokens (:obj:`int`): the number of tokens in the beginning
            of the sequence to keep unchanged from sampling. This is useful when
            special tokens have been added to the beginning of the sequence.
            Default to 1.
        data_style (:obj:`str`): the style of the data. If "pcpt", the data is
            masked and padded for perception training. If "gen", only the gene
            tokens are provided, but not the expression values, for pure generative
            training setting. If "both", the output will contain both fields above.
            Choices: "pcpt", "gen", "both". Default to "pcpt".
    """
    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    tf_experimental_compile: bool = False
    return_tensors: str = "pt"
    
    do_padding: bool = True
    pad_token_id: int = 0
    pad_value: int = 0
    do_mlm: bool = True
    do_binning: bool = True
    mask_value: int = -1
    max_length: int = 512
    sampling: bool = True
    #reserve_keys: List[str] = field(default_factory=lambda: [])
    reserve_keys: List[str] = []
    keep_first_n_tokens: int = 1
    data_style: str = "pcpt"
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        mlm: bool = True,
        mlm_probability: float = 0.15,
        pad_to_multiple_of: Optional[int] = None,
        tf_experimental_compile: bool = False,
        return_tensors: str = "pt",
        do_padding: bool = True,
        pad_token_id: Optional[int] = None,
        pad_value: int = 0,
        do_mlm: bool = True,
        do_binning: bool = True,
        mask_value: int = -1,
        max_length: Optional[int] = None,
        sampling: bool = True,
        reserve_keys: List[str] = [],
        #reserve_keys: List[str] = field(default_factory=lambda: []),
        keep_first_n_tokens: int = 1,
        data_style: str = "pcpt",
    ):
        super().__init__(tokenizer, 
                         mlm=mlm, 
                         mlm_probability=mlm_probability, 
                         pad_to_multiple_of=pad_to_multiple_of, 
                         tf_experimental_compile=tf_experimental_compile,
                         return_tensors=return_tensors,
                        )
        self.do_padding = do_padding
        self.pad_token_id = pad_token_id
        self.pad_value = pad_value
        self.do_mlm = do_mlm 
        self.do_binning = do_binning
        self.mask_value = mask_value
        self.max_length = max_length
        self.sampling = sampling
        self.reserve_keys = reserve_keys
        self.keep_first_n_tokens = keep_first_n_tokens
        self.data_style = data_style

    def __post_init__(self):
        if self.do_padding:
            if self.pad_token_id is None:
                raise ValueError("`pad_token_id` is required if `do_padding`.")
            if self.max_length is None:
                raise ValueError("`max_length` is required if `do_padding`.")

        if isinstance(self.mlm_probability, float):
            if self.mlm_probability <= 0 or self.mlm_probability >= 1:
                raise ValueError("`mlm_probability` must be between 0 and 1.")
        elif isinstance(self.mlm_probability, (list, tuple)):
            if min(self.mlm_probability) <= 0 or max(self.mlm_probability) >= 1:
                raise ValueError("`mlm_probability` must be between 0 and 1.")
        else:
            raise ValueError("`mlm_probability` must be a float or iterable of floats.")

        if isinstance(self.reserve_keys, str):
            self.reserve_keys = [self.reserve_keys]

        if self.keep_first_n_tokens < 0 or self.keep_first_n_tokens > self.max_length:
            raise ValueError(
                "`keep_first_n_tokens` must be between 0 and `max_length` "
                f"({self.max_length})."
            )

        if self.data_style not in ["pcpt", "gen", "both"]:
            raise ValueError("`data_style` must be one of 'pcpt', 'gen', 'both'.")

    def __call__(
        self, examples: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            examples (:obj:`List[Dict[str, torch.Tensor]]`): a list of data dicts.
                Each dict is for one cell. It contains multiple 1 dimensional tensors
                like the following exmaple:
                    {'id': tensor(184117),
                    'genes': tensor([36572, 17868, ..., 17072]),
                    'expressions': tensor([ 0.,  2., ..., 18.])}

        Returns:
            :obj:`Dict[str, torch.Tensor]`: a dict of tensors.
        """
        
        if len(self.reserve_keys) > 0:
            assert all(key in examples[0] for key in self.reserve_keys), (
                f"reserve_keys must be a subset of the keys in the examples. "
                f"Got {self.reserve_keys} but expected keys in {list(examples[0].keys())}."
            )

        if self.data_style == "pcpt":
            data_dict = self._call_pcpt(examples)
        elif self.data_style == "gen":
            data_dict = self._call_gen(examples)
        elif self.data_style == "both":
            data_dict = self._call_both(examples)

        # add reserved keys
        device = examples[0]["genes"].device
        for key in self.reserve_keys:
            data_ = [example[key] for example in examples]
            data_dict[key] = torch.stack(data_, dim=0).to(device)

        return data_dict

    def _call_pcpt(
        self, examples: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Each example is like:
            {'id': tensor(184117),
            'genes': tensor([36572, 17868, ..., 17072]),
            'expressions': tensor([ 0.,  2., ..., 18.])}

        Args:
            examples (:obj:`List[Dict[str, torch.Tensor]]`): a list of examples.
                Each example is a dictionary of tensors.
        Returns:
            :obj:`Dict[str, torch.Tensor]`: a dictionary of tensors.
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
            masked_expressions = self._mask(
                padded_expressions, self.keep_first_n_tokens
            )
        else:
            masked_expressions = padded_expressions
        data_dict["masked_expr"] = masked_expressions

        return data_dict

    def _call_gen(
        self, examples: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        This method will simply return the gene ids, with needed padding. There is
        no masking for pure generative training, and no input of expr values.

        Each example is like:
            {'id': tensor(184117),
            'genes': tensor([36572, 17868, ..., 17072])}

        Returns:
            Dict[str, torch.Tensor]: a dict of tensors.
            Example:
                {'pcpt_gene': tensor([[36572, 17868, ..., 17072],
                                        [36572, 17868, ..., 17072],
                                        ...,
                                        [36572, 17868, ..., 17072]]),
                'pcpt_expr': tensor([[ 0.,  2., ..., 18.],
                                        [ 0.,  2., ..., 18.],
                                        ...,
                                        [ 0.,  2., ..., 18.]])}
        """

        if not isinstance(examples[0], Mapping):
            return NotImplementedError

        device = examples[0]["genes"].device

        max_ori_len = max(len(example["genes"]) for example in examples)
        _max_length = self.max_length if max_ori_len >= self.max_length else max_ori_len

        # pad and truncate
        padded_pcpt_genes = []
        padded_pcpt_expressions = []
        for i in range(len(examples)):
            genes = examples[i]["genes"]
            expressions = examples[i]["expressions"]
            if self.do_binning:
                expressions[self.keep_first_n_tokens :] = binning(
                    row=expressions[self.keep_first_n_tokens :],
                    n_bins=51,  # FIXME: replace with self.n_bins
                )
            genes, expressions = self._sample_or_truncate_plus_pad(
                genes, expressions, _max_length
            )
            padded_pcpt_genes.append(genes)
            padded_pcpt_expressions.append(expressions)

        padded_pcpt_genes = torch.stack(padded_pcpt_genes, dim=0).to(device)
        padded_pcpt_expressions = torch.stack(padded_pcpt_expressions, dim=0).to(device)

        data_dict = {
            "pcpt_gene": padded_pcpt_genes,
            "pcpt_expr": padded_pcpt_expressions,
        }
        return data_dict

    def _call_both(
        self,
        examples: List[Dict[str, torch.Tensor]],
        gen_prob: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        This method will split the input into the peception part and the generation
        part. The perception part will be processed into gene ids and expr values,
        and the generation part will be processed into gene ids only.

        By default, the mlm_probability will be used to select the genese assigned to
        the generation part.

        Each example is like:
            {'id': tensor(184117),
            'genes': tensor([36572, 17868, ..., 17072]),
            'expressions': tensor([ 0.,  2., ..., 18.])}

        Args:
            gen_prob (float, optional): the probability of a gene being assigned to
                the generation part. If not provided, the mlm_probability will be used.

        Returns:
            Dict[str, torch.Tensor]: a dict of tensors.
            Example:
                {'pcpt_gene': tensor([[36572, 17868, ..., 17072],
                                        [36572, 17868, ..., 17072],
                                        ...,
                                        [36572, 17868, ..., 17072]]),
                'pcpt_expr': tensor([[ 0.,  2., ..., 18.],
                                        [ 0.,  2., ..., 18.],
                                        ...,
                                        [ 0.,  2., ..., 18.]]),
                'gen_gene': tensor([[36573, 17869, ..., 17073],
                                        [36573, 17869, ..., 17073],
                                        ...,
                                        [36573, 17869, ..., 17073]]),
                'gen_expr_target': tensor([[ 1.,  3., ..., 19.],
                                        [ 1.,  3., ..., 19.],
                                        ...,
                                        [ 1.,  3., ..., 19.]])}
        """
        if not isinstance(examples[0], Mapping):
            return NotImplementedError

        if not self.do_mlm:
            # if not doing mlm, then the perceptrual part is the whole input
            return self._call_gen(examples)

        if gen_prob is None:
            gen_prob = self.get_mlm_probability()

        device = examples[0]["genes"].device

        max_ori_len = max(len(example["genes"]) for example in examples)
        _max_length = self.max_length if max_ori_len >= self.max_length else max_ori_len

        gen_length = int((_max_length - self.keep_first_n_tokens) * gen_prob)
        pcpt_length = _max_length - gen_length  # perception part length

        # pad and truncate
        padded_pcpt_genes = []
        padded_pcpt_expressions = []
        padded_gen_genes = []
        padded_gen_expressions = []
        for i in range(len(examples)):
            genes = examples[i]["genes"]
            expressions = examples[i]["expressions"]
            if self.do_binning:
                expressions[self.keep_first_n_tokens :] = binning(
                    row=expressions[self.keep_first_n_tokens :],
                    n_bins=51,
                )

            (
                gen_genes,
                gen_expressions,
                pcpt_genes,
                pcpt_expressions,
            ) = self._random_split(
                genes[self.keep_first_n_tokens :],
                expressions[self.keep_first_n_tokens :],
                ratio=gen_prob,
            )
            pcpt_genes = torch.cat(
                (genes[: self.keep_first_n_tokens], pcpt_genes), dim=0
            )
            pcpt_expressions = torch.cat(
                (expressions[: self.keep_first_n_tokens], pcpt_expressions), dim=0
            )

            pcpt_genes, pcpt_expressions = self._sample_or_truncate_plus_pad(
                pcpt_genes, pcpt_expressions, pcpt_length
            )  # torch tensors of length pcpt_length
            padded_pcpt_genes.append(pcpt_genes)
            padded_pcpt_expressions.append(pcpt_expressions)

            gen_genes, gen_expressions = self._sample_or_truncate_plus_pad(
                gen_genes, gen_expressions, gen_length
            )  # torch tensors of length gen_length
            padded_gen_genes.append(gen_genes)
            padded_gen_expressions.append(gen_expressions)

        padded_pcpt_genes = torch.stack(padded_pcpt_genes, dim=0)
        padded_pcpt_expressions = torch.stack(padded_pcpt_expressions, dim=0)
        padded_gen_genes = torch.stack(padded_gen_genes, dim=0)
        padded_gen_expressions = torch.stack(padded_gen_expressions, dim=0)

        data_dict = {
            "pcpt_gene": padded_pcpt_genes,
            "pcpt_expr": padded_pcpt_expressions,
            "gen_gene": padded_gen_genes,
            "gen_expr_target": padded_gen_expressions,
        }

        return data_dict

    def _random_split(
        self,
        *arrays: torch.Tensor,
        ratio: float,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Randomly split the arrays into two parts. The first part will have the
        length of `ratio * length`, and the second part will have the length of
        `(1 - ratio) * length`. When multiple arrays are provided, they are supposed
        to have the same length.

        This method reflects the behavior of `sklearn.model_selection.train_test_split`

        Args:
            *arrays (torch.Tensor): the arrays to be split.
            ratio (float): the ratio of the first part.

        Returns:
            Tuple[torch.Tensor, ...]: the split arrays.
        """
        assert len(arrays) > 0
        assert 0 < ratio < 1
        if len(arrays) > 1:
            assert all(
                array.shape[0] == arrays[0].shape[0] for array in arrays
            ), "The arrays must have the same length."

        length = arrays[0].shape[0]
        split_index = int(length * ratio)

        indices = torch.randperm(length, device=arrays[0].device)
        first_part_indices = indices[:split_index]
        second_part_indices = indices[split_index:]

        first_parts = tuple(array[first_part_indices] for array in arrays)
        second_parts = tuple(array[second_part_indices] for array in arrays)

        return first_parts + second_parts

    def get_mlm_probability(self) -> float:
        """
        Get the mlm probability for the current step.
        """
        if isinstance(self.mlm_probability, float):
            return self.mlm_probability
        elif isinstance(self.mlm_probability, list):
            # random choose a probability
            return np.random.choice(self.mlm_probability)
        else:
            raise ValueError(
                "mlm_probability must be a float or a list of floats, "
                f"but got {self.mlm_probability}."
            )

    def _mask(
        self, expressions: torch.Tensor, keep_first_n_tokens: int = 0
    ) -> torch.Tensor:
        """
        Mask the expression values with MLM.
        """
        if keep_first_n_tokens > 0:
            result_ = self._mask(
                expressions[:, keep_first_n_tokens:],
                keep_first_n_tokens=0,
            )
            return torch.cat([expressions[:, :keep_first_n_tokens], result_], dim=1)

        device = expressions.device
        shape = expressions.shape

        probability_matrix = torch.full(shape, self.get_mlm_probability())
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
