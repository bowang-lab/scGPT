from typing import Sequence

import numpy as np
import torch

from scgpt.data_sampler import SubsetSequentialSampler, SubsetsBatchSampler
from scgpt.utils import set_seed

set_seed(1)


def _check_reorder(a: Sequence[int], b: Sequence[int]) -> bool:
    """Check if a is a shuffled version of b."""
    if len(a) != len(b):
        return False
    if len(a) == 0:
        return True
    if len(a) == 1:
        return a[0] == b[0]

    # Find the first element in a that is also in b
    first = a[0]
    for i in range(len(b)):
        if b[i] == first:
            break
    else:
        return False

    # Check if the rest of a is a shuffled version of b[:i] + b[i+1:]
    return _check_reorder(a[1:], b[:i] + b[i + 1 :])


def test_subset_sequential_sampler() -> None:
    """Test SubsetSequentialSampler."""
    # Test with a list of indices
    sampler = SubsetSequentialSampler([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert list(sampler) == list(range(10))


def test_subsets_batch_sampler() -> None:
    """Test SubsetsBatchSampler."""
    # Test with a list of subsets of indices
    subsets = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]

    # Test sequential sampling
    sampler = SubsetsBatchSampler(
        subsets,
        batch_size=3,
        intra_subset_shuffle=False,
        inter_subset_shuffle=False,
    )
    assert list(sampler) == [[0, 1, 2], [3, 4], [5, 6, 7], [8, 9]]

    # Test sequential sampling with drop_last
    sampler = SubsetsBatchSampler(
        subsets,
        batch_size=3,
        intra_subset_shuffle=False,
        inter_subset_shuffle=False,
        drop_last=True,
    )
    assert list(sampler) == [[0, 1, 2], [5, 6, 7]]

    # Test sampling with inter subset shuffle
    sampler = SubsetsBatchSampler(
        subsets,
        batch_size=3,
        intra_subset_shuffle=False,
        inter_subset_shuffle=True,
    )
    sampled_idx = list(sampler)
    assert len(sampled_idx) == 4
    print(sampled_idx)
    assert sampled_idx != [[0, 1, 2], [3, 4], [5, 6, 7], [8, 9]]
    assert _check_reorder(sampled_idx, [[0, 1, 2], [3, 4], [5, 6, 7], [8, 9]])

    # Test sampling with intra subset shuffle
    sampler = SubsetsBatchSampler(
        subsets,
        batch_size=3,
        intra_subset_shuffle=True,
        inter_subset_shuffle=False,
    )
    sampled_idx = list(sampler)
    assert len(sampled_idx) == 4
    sampled_idx_subset0 = sampled_idx[0] + sampled_idx[1]
    sampled_idx_subset1 = sampled_idx[2] + sampled_idx[3]
    assert sampled_idx_subset0 != [0, 1, 2, 3, 4]
    assert sampled_idx_subset1 != [5, 6, 7, 8, 9]
    assert _check_reorder(sampled_idx_subset0, [0, 1, 2, 3, 4])
    assert _check_reorder(sampled_idx_subset1, [5, 6, 7, 8, 9])

    # Test sampling with both intra and inter subset shuffle
    sampler = SubsetsBatchSampler(
        subsets,
        batch_size=3,
        intra_subset_shuffle=True,
        inter_subset_shuffle=True,
    )
    sampled_idx = list(sampler)
    assert len(sampled_idx) == 4
    sampled_idx_flatten = [item for sublist in sampled_idx for item in sublist]
    assert sampled_idx_flatten != [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert _check_reorder(sampled_idx_flatten, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
