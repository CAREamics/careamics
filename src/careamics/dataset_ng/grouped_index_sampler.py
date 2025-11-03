"""Module for the `GroupedIndexSampler`."""

from collections.abc import Iterator, Sequence
from typing import Self

import numpy as np
from numpy.random import Generator, default_rng
from torch.utils.data import Sampler

from careamics.dataset_ng.dataset import CareamicsDataset


class GroupedIndexSampler(Sampler):
    """
    A PyTorch Sampler iterates through groups of indices.

    The order of the groups will be shuffled and the order of the indices within the
    groups will be shuffled.

    This sampler is useful for iterative file loading — one file should be loaded at a
    time so indices belonging to the same file should be grouped, but the order of the
    files and the order of the indices should be shuffled.
    """

    def __init__(self, grouped_indices: Sequence[Sequence[int]], rng: Generator | None):
        """
        Parameters
        ----------
        grouped_indices : Sequence of (Sequence of int)
            The indices that should be iterated through in groups.
        """
        super().__init__()
        if rng is None:
            self.rng = default_rng()
        else:
            self.rng = rng
        # TODO: validate indices are unique across groups
        self.grouped_indices = grouped_indices

    @classmethod
    def from_dataset(
        cls, dataset: CareamicsDataset, rng: Generator | None = None
    ) -> Self:
        """
        Create the sampler from a CareamicsDataset.

        The grouped indices will be retrieved from the dataset's patching strategy.

        Parameters
        ----------
        dataset: CareamicsDataset
            An instance of the CareamicsDataset to create the sampler for.
        rng: numpy.random.Generator, optional
            Numpy random number generator that can be used to seed the sampler.
        """
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
