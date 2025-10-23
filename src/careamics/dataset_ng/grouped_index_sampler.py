from collections.abc import Iterator, Sequence
from typing import Self

import numpy as np
from numpy.random import Generator, default_rng
from torch.utils.data import Sampler

from careamics.dataset_ng.dataset import CareamicsDataset


class GroupedIterSampler(Sampler):

    def __init__(self, grouped_indices: Sequence[Sequence[int]], rng: Generator | None):
        super().__init__()
        if rng is None:
            self.rng = default_rng()
        else:
            self.rng = rng
        self.grouped_indices = grouped_indices

    @classmethod
    def from_dataset(cls, dataset: CareamicsDataset, rng: Generator | None) -> Self:
        n_data_samples = len(dataset.input_extractor.shape)
        grouped_indices: list[Sequence[int]] = [
            dataset.patching_strategy.get_patch_indices(i)
            for i in range(n_data_samples)
        ]
        return cls(grouped_indices=grouped_indices, rng=rng)

    def __iter__(self) -> Iterator[int]:

        # shuffle the groups and the sub groups but keep indices in a group adjacent
        group_order = np.arange(len(self.grouped_indices))
        self.rng.shuffle(group_order)
        for group_idx in group_order:
            group = self.grouped_indices[group_idx.item()]
            index_order = np.arange(len(group))
            self.rng.shuffle(index_order)
            for idx in index_order:
                yield group[idx.item()]
