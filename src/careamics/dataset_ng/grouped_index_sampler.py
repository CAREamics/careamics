"""Module for the `GroupedIndexSampler`."""

from collections.abc import Iterator, Sequence
from typing import Self

import numpy as np
from numpy.random import Generator, default_rng
from torch.utils.data import Sampler

from careamics.dataset_ng.dataset import CareamicsDataset


class GroupedIndexSampler(Sampler):
    """A PyTorch Sampler that iterates through groups of indices.

    The order of the groups and the order of indices within each group are shuffled.

    This sampler is useful for iterative file loading — one file should be loaded at a
    time so indices belonging to the same file should be grouped, but the order of the
    files and the order of the indices should be shuffled.

    Parameters
    ----------
    grouped_indices : Sequence of (Sequence of int)
        The indices to iterate through, grouped (e.g. by file).
    rng : numpy.random.Generator or None
        Random number generator for shuffling. If None, a default generator is used.
    """

    def __init__(self, grouped_indices: Sequence[Sequence[int]], rng: Generator | None):
        """Initialize the sampler from grouped index sequences.

        Parameters
        ----------
        grouped_indices : Sequence of (Sequence of int)
            The indices to iterate through, grouped (e.g. by file).
        rng : numpy.random.Generator or None
            Random number generator for shuffling. If None, a default generator is used.
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
        """Create the sampler from a CareamicsDataset.

        The grouped indices will be retrieved from the dataset's patching strategy.

        Parameters
        ----------
        dataset : CareamicsDataset
            An instance of the CareamicsDataset to create the sampler for.
        rng : numpy.random.Generator, optional
            Random number generator used to seed the sampler. If None, a default
            generator is used.

        Returns
        -------
        GroupedIndexSampler
            A sampler yielding indices grouped by the dataset's patching strategy.
        """
        n_data_samples = len(dataset.input_extractor.shapes)
        grouped_indices: list[Sequence[int]] = [
            dataset.patching_strategy.get_patch_indices(i)
            for i in range(n_data_samples)
        ]
        return cls(grouped_indices=grouped_indices, rng=rng)

    def __iter__(self) -> Iterator[int]:
        """Iterate over indices with groups and within-group order shuffled.

        Returns
        -------
        Iterator[int]
            Indices from all groups in shuffled group order and shuffled order
            within each group.
        """
        # shuffle the groups and the sub groups but keep indices in a group adjacent
        group_order = np.arange(len(self.grouped_indices))
        self.rng.shuffle(group_order)
        for group_idx in group_order:
            group = self.grouped_indices[group_idx.item()]
            index_order = np.arange(len(group))
            self.rng.shuffle(index_order)
            for idx in index_order:
                yield group[idx.item()]
