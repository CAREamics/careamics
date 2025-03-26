from typing import TYPE_CHECKING, Optional

import numpy as np
from torch.utils.data import BatchSampler

if TYPE_CHECKING:
    from ..dataset import CareamicsDataset
    from ..patch_extractor.image_stack import ManagedLazyImageStack


# TODO: decouple from CAREamicsDataset by having n_files as argument
#   Also have to give patch indices as argument
class GroupedBatchSampler(BatchSampler):

    def __init__(
        self,
        dataset: "CareamicsDataset[ManagedLazyImageStack]",
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
        max_files_loaded: int = 1,
        random_seed: Optional[int] = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.max_files_loaded = max_files_loaded

        self.rng = np.random.default_rng(seed=random_seed)

    # TODO: separate code into more manageable functions
    def __iter__(self):

        n_files = len(self.dataset.input_extractor.image_stacks)
        file_indices = np.arange(n_files)
        if self.shuffle:
            self.rng.shuffle(file_indices)

        # put files into groups max_files_loaded
        file_groups = [
            file_indices[i : i + self.max_files_loaded]
            for i in range(0, n_files, self.max_files_loaded)
        ]
        remaining_indices = None  # used to concat remaining files of previous files
        for file_group in file_groups:
            # get all the corresponding patch indices of the files in the file group
            patch_indices = np.zeros((0,))
            for file_index in file_group:
                file_patch_indices = self.dataset.patching_strategy.get_patch_indices(
                    file_index
                )
                patch_indices = np.concatenate([patch_indices, file_patch_indices])
            if self.shuffle:
                self.rng.shuffle(patch_indices)

            # concat remaining indices from previous batch at the start
            if remaining_indices is not None:
                patch_indices = np.concatenate([remaining_indices, patch_indices])
            remaining_indices = None

            # yield batches, if not equal to batch size, it is set to remaining indices
            for i in range(0, len(patch_indices), self.batch_size):
                batch_indices = (
                    (patch_indices[i : i + self.batch_size]).astype(int).tolist()
                )
                if len(batch_indices) == self.batch_size:
                    yield batch_indices
                else:
                    remaining_indices = batch_indices

        # remove
        if remaining_indices is not None:
            if not self.drop_last:
                yield remaining_indices

    # TODO: need to make sure all remaining files are closed at the end
