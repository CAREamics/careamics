from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
from torch.utils.data import BatchSampler

if TYPE_CHECKING:
    from ..dataset import CareamicsDataset
    from ..patch_extractor.image_stack import ManagedLazyImageStack


class FifoBatchSampler(BatchSampler):

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
                batch_indices = list(patch_indices[i : i + self.batch_size])
                if len(batch_indices) == self.batch_size:
                    yield batch_indices
                else:
                    remaining_indices = batch_indices

        # remove
        if remaining_indices is not None:
            if not self.drop_last:
                yield remaining_indices

    # TODO: need to make sure all remaining files are closed at the end


class FifoImageStackManager:

    def __init__(self, max_files_loaded: int = 1):
        self.image_stacks: dict[Path, ManagedLazyImageStack] = {}
        self.loaded_image_stacks: list[ManagedLazyImageStack] = []
        self.max_files_loaded = max_files_loaded

    @property
    def currently_loaded(self):
        return len(self.loaded_image_stacks)

    def register_image_stack(self, image_stack: "ManagedLazyImageStack"):
        self.image_stacks[image_stack.path] = image_stack
        image_stack.set_callbacks(
            on_load=self.notify_load,
            on_close=self.notify_close,
        )

    def register_image_stacks(self, image_stacks: Sequence["ManagedLazyImageStack"]):
        for image_stack in image_stacks:
            self.register_image_stack(image_stack)

    def free(self):
        if self.currently_loaded >= self.max_files_loaded:
            image_stack_to_close = self.loaded_image_stacks[0]  # FIFO
            image_stack_to_close.deallocate()

    def notify_load(self, image_stack: "ManagedLazyImageStack"):
        if image_stack.path not in self.image_stacks:
            raise ValueError(
                f"Image stack with path {image_stack.path} has not been registered."
            )
        # TODO: Raise error if already loaded?
        self.free()
        self.loaded_image_stacks.append(image_stack)

    def notify_close(self, image_stack: "ManagedLazyImageStack"):
        if image_stack.path not in self.image_stacks:
            raise ValueError(
                f"Image stack with path {image_stack.path} has not been registered."
            )
        loaded_paths = [img_stack.path for img_stack in self.loaded_image_stacks]
        file_id = loaded_paths.index(image_stack.path)  # TODO: this will be file path
        self.loaded_image_stacks.pop(file_id)

    def close_all(self):
        loaded_image_stacks = self.loaded_image_stacks.copy()
        for image_stack in loaded_image_stacks:
            image_stack.deallocate()
