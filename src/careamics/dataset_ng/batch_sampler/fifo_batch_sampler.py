from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
from torch.utils.data import BatchSampler

if TYPE_CHECKING:
    from ..dataset import CareamicsDataset
    from ..patch_extractor.image_stack import ManagedLazyImageStack


# TODO: Could decouple this from the CAREamics dataset
#   need to provide patch indices per file


class FifoBatchSampler(BatchSampler):
    """
    Produces batch indices designed to work with a first-in-first-out loading system.

    It ensures that the image stacks are loaded and closed following a fifo strategy.
    """

    def __init__(
        self,
        dataset: "CareamicsDataset[ManagedLazyImageStack]",
        # batch_size, shuffle and drop last would normally go to the dataloader but
        #   are mutually exclusive with a custom batchsampler
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
        max_files_loaded: int = 2,
        random_seed: Optional[int] = None,
    ):
        """
        Produces indices designed to work with a first-in-first-out loading system.

        Parameters
        ----------
        dataset : CareamicsDataset[ManagedLazyImageStack]
            The dataset to produce batch indices for. The underlying image stacks must
            be instances of the `ManagedLazyImageStack`.
        batch_size : int
            The batch size.
        shuffle : bool, default=False
            Whether the data should be randomly shuffled.
        drop_last : bool, optional, default=False
            Set to True to drop the last incomplete batch, if the dataset size is not
            divisible by the batch size. If False and the size of dataset is not
            divisible by the batch size, then the last batch will be smaller.
        max_files_loaded : int, default=2
            The maximum number of files that should have their data loaded at one time.
        random_seed : int, optional
            A random seed to seed the shuffling if it is shuffled.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.max_files_loaded = max_files_loaded
        self.rng = np.random.default_rng(seed=random_seed)
        self.fifo_manager = self._init_fifo_manager()

    # TODO: move reference to fifo manager out of BatchSampler
    #   BatchSampler's responsibility should really only be to produce the batch indices
    #   But not sure where else to put it, in PatchExtractor?
    #   Putting it in the patch extractor will be annoying if we end up having more
    #   than one patch extractor like the LC extractor.
    def _init_fifo_manager(self) -> "FifoImageStackManager":
        fifo_manager = FifoImageStackManager(max_files_loaded=self.max_files_loaded)
        fifo_manager.register_image_stacks(self.dataset.input_extractor.image_stacks)
        if self.dataset.target_extractor is not None:
            fifo_manager.register_image_stacks(
                self.dataset.target_extractor.image_stacks
            )
        return fifo_manager

    # TODO: separate code into more manageable functions
    def __iter__(self):

        n_files = len(self.dataset.input_extractor.image_stacks)
        file_indices = np.arange(n_files)
        if self.shuffle:
            self.rng.shuffle(file_indices)

        # put files into groups of size max_files_loaded
        file_groups = [
            file_indices[i : i + self.max_files_loaded]
            for i in range(0, n_files, self.max_files_loaded)
        ]
        remaining_indices = None  # used to concat remaining indices of previous file
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

        # return last remaining indices (if not drop_last)
        if remaining_indices is not None:
            if not self.drop_last:
                yield remaining_indices

        # close remaining image stacks at the end of iteration
        self.fifo_manager.close_all()
        # NOTE: if fifo_manager is moved out of BatchSampler will have to find another
        #   way to close remaining images


class FifoImageStackManager:
    """
    Manages closing `ManagedLazyImageStack`s using the First-in-first-out strategy.

    This class will make sure there are never more loaded-image-stacks than the
    specified `max_files_loaded`. It does this by closing the least recently opened
    image stack.
    """

    def __init__(self, max_files_loaded: int = 2):
        """
        Manages closing `ManagedLazyImageStack`s using the First-in-first-out strategy.

        Parameters
        ----------
        max_files_loaded : int, default=2
            The maximum number of files that should have their data loaded at one time.
        """
        if max_files_loaded < 1:
            raise ValueError(
                "Maximum number of image stacks loaded must be greater than zero."
            )
        # the dict of all registered image stacks
        #   image stacks should be registered before the manager can manage them
        #   see `register_image_stack` method.
        self.image_stacks: dict[Path, ManagedLazyImageStack] = {}
        # A list of the loaded image stacks, identified by their path.
        self.loaded_image_stacks: list[ManagedLazyImageStack] = []
        self.max_files_loaded = max_files_loaded

    @property
    def n_currently_loaded(self):
        """The number of image stacks currently loaded."""
        return len(self.loaded_image_stacks)

    def register_image_stack(self, image_stack: "ManagedLazyImageStack"):
        """
        Register an image stack to the `FifoImageStackManager`.

        Parameters
        ----------
        image_stack : ManagedLazyImageStack
            The image stack to register.

        Raises
        ------
        ValueError
            If the image stack has already been registered.
        """
        if image_stack.path in self.image_stacks:
            raise ValueError(
                "Image stack has already been registered to the "
                f"`FifoImageStackManager`, image stack path: {image_stack.path}."
            )
        self.image_stacks[image_stack.path] = image_stack
        image_stack.set_callbacks(
            on_load=self.notify_load,
            on_close=self.notify_close,
        )

    def register_image_stacks(self, image_stacks: Sequence["ManagedLazyImageStack"]):
        """
        Register multiple image stacks to the `FifoImageStackManager`.

        Parameters
        ----------
        image_stacks : sequence of ManagedLazyImageStack
            A sequence of image stacks to register to the `FifoImageStackManager`.
        """
        for image_stack in image_stacks:
            self.register_image_stack(image_stack)

    def free(self):
        """
        Free space to allow a new image stack to be opened.

        This is achieved by closing the image stack that was opened least recently.
        """
        if self.n_currently_loaded >= self.max_files_loaded:
            image_stack_to_close = self.loaded_image_stacks[0]  # FIFO
            image_stack_to_close.deallocate()  # this will in turn call notify_close

    def notify_load(self, image_stack: "ManagedLazyImageStack"):
        """Notify the `FifoImageStackManager that the `image_stack` has been loaded.

        Parameters
        ----------
        image_stack : ManagedLazyImageStack
            The image stack that has been loaded

        Raises
        ------
        ValueError
            If the `image_stack` has not been registered.
        """
        if image_stack.path not in self.image_stacks:
            raise ValueError(
                f"Image stack with path {image_stack.path} has not been registered."
            )
        # TODO: Raise error if already loaded?
        self.free()
        self.loaded_image_stacks.append(image_stack)

    def notify_close(self, image_stack: "ManagedLazyImageStack"):
        """Notify the `FifoImageStackManager that the `image_stack` has been closed.

        Parameters
        ----------
        image_stack : ManagedLazyImageStack
            The image stack that has been closed.

        Raises
        ------
        ValueError
            If the `image_stack` has not been registered.
        """
        if image_stack.path not in self.image_stacks:
            raise ValueError(
                f"Image stack with path {image_stack.path} has not been registered."
            )
        loaded_paths = [img_stack.path for img_stack in self.loaded_image_stacks]
        file_id = loaded_paths.index(image_stack.path)
        self.loaded_image_stacks.pop(file_id)

    def close_all(self):
        """Close all the currently loaded image stacks."""
        loaded_image_stacks = self.loaded_image_stacks.copy()
        for image_stack in loaded_image_stacks:
            image_stack.deallocate()
