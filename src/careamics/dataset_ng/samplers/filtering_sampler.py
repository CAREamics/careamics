from collections.abc import Iterator

import numpy as np
from torch.utils.data import Sampler

from careamics.dataset_ng.dataset import CareamicsDataset


class FilteringSampler(Sampler):
    """
    A sampler that filters patches based on coordinate and value filters.

    This sampler pre-filters patches using coordinate filters and applies
    value-based filters with a patience mechanism during sampling.
    """

    def __init__(
        self,
        dataset: CareamicsDataset,
        patch_filter=None,
        coord_filter=None,
        patience: int = 10,
        seed: int | None = None,
    ):
        self.dataset = dataset
        self.patch_filter = patch_filter
        self.coord_filter = coord_filter
        self.patience = patience
        self.rng = np.random.default_rng(seed)

        # Pre-filter using coordinate filter
        self._valid_indices = self._precompute_coord_filtered_indices()

    def _precompute_coord_filtered_indices(self) -> list[int]:
        """Pre-filter indices using coordinate filter."""
        if self.coord_filter is None:
            return list(range(len(self.dataset)))

        valid_indices = []
        for idx in range(len(self.dataset)):
            patch_spec = self.dataset.patching_strategy.get_patch_spec(idx)
            if not self.coord_filter.filter_out(patch_spec):
                valid_indices.append(idx)

        return valid_indices

    def __iter__(self) -> Iterator[int]:
        """Generate indices with patch filtering applied."""
        indices = self.rng.permutation(self._valid_indices).tolist()

        if self.patch_filter is None:
            yield from indices
            return

        for idx in indices:
            patience = self.patience
            while patience > 0:
                patch_spec = self.dataset.patching_strategy.get_patch_spec(idx)
                patch = self.dataset.input_extractor.extract_patch(
                    data_idx=patch_spec["data_idx"],
                    sample_idx=patch_spec["sample_idx"],
                    coords=patch_spec["coords"],
                    patch_size=patch_spec["patch_size"],
                )

                if not self.patch_filter.filter_out(patch):
                    yield idx
                    break

                patience -= 1
                idx = self.rng.choice(self._valid_indices)  # Try another random index
            else:
                yield idx

    def __len__(self) -> int:
        return len(self._valid_indices)
