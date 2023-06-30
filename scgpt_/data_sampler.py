from typing import Iterable, List, Sequence

import numpy as np
import torch
from torch.utils.data import Sampler, SubsetRandomSampler, BatchSampler


class SubsetSequentialSampler(Sampler):
    """Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices: Sequence[int]):
        self.indices = indices

    def __iter__(self) -> Iterable[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


class SubsetsBatchSampler(Sampler[List[int]]):
    r"""Samples batches of indices from a list of subsets of indices. Each subset
    of indices represents a data subset and is sampled without replacement randomly
    or sequentially. Specially, each batch only contains indices from a single subset.
    This sampler is for the scenario where samples need to be drawn from multiple
    subsets separately.

    Arguments:
        subsets (List[Sequence[int]]): A list of subsets of indices.
        batch_size (int): Size of mini-batch.
        intra_subset_shuffle (bool): If ``True``, the sampler will shuffle the indices
            within each subset.
        inter_subset_shuffle (bool): If ``True``, the sampler will shuffle the order
            of subsets.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``.
    """

    def __init__(
        self,
        subsets: List[Sequence[int]],
        batch_size: int,
        intra_subset_shuffle: bool = True,
        inter_subset_shuffle: bool = True,
        drop_last: bool = False,
    ):
        self.subsets = subsets
        self.batch_size = batch_size
        self.intra_subset_shuffle = intra_subset_shuffle
        self.inter_subset_shuffle = inter_subset_shuffle
        self.drop_last = drop_last

        if intra_subset_shuffle:
            self.subset_samplers = [SubsetRandomSampler(subset) for subset in subsets]
        else:
            self.subset_samplers = [
                SubsetSequentialSampler(subset) for subset in subsets
            ]

        self.batch_samplers = [
            BatchSampler(sampler, batch_size, drop_last)
            for sampler in self.subset_samplers
        ]

        if inter_subset_shuffle:
            # maintain a mapping from sample batch index to batch sampler
            _id_to_batch_sampler = []
            for i, batch_sampler in enumerate(self.batch_samplers):
                _id_to_batch_sampler.extend([i] * len(batch_sampler))
            self._id_to_batch_sampler = np.array(_id_to_batch_sampler)

            assert len(self._id_to_batch_sampler) == len(self)

            self.batch_sampler_iterrators = [
                batch_sampler.__iter__() for batch_sampler in self.batch_samplers
            ]

    def __iter__(self) -> Iterable[List[int]]:
        if self.inter_subset_shuffle:
            # randomly sample from batch samplers
            random_idx = torch.randperm(len(self._id_to_batch_sampler))
            batch_sampler_ids = self._id_to_batch_sampler[random_idx]
            for batch_sampler_id in batch_sampler_ids:
                batch_sampler_iter = self.batch_sampler_iterrators[batch_sampler_id]
                yield next(batch_sampler_iter)
        else:
            for batch_sampler in self.batch_samplers:
                yield from batch_sampler

    def __len__(self) -> int:
        return sum(len(batch_sampler) for batch_sampler in self.batch_samplers)
