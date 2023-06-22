import os
import itertools
import logging
import tifffile
import torch

import zarr
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Callable, Generator, List, Optional, Tuple, Union

from . import (
    compute_overlap,
    compute_reshaped_view,
    compute_patch_steps,
    are_axes_valid,
    compute_crop_and_stitch_coords_1d,
)

from ..utils import normalize

############################################
#   ETL pipeline
############################################


logger = logging.getLogger(__name__)


class PatchDataset(torch.utils.data.IterableDataset):
    """Dataset to extract patches from a list of images and apply transforms to the patches."""

    def __init__(
        self,
        data_path: str,
        ext: str,
        axes: str,
        num_files: int,
        data_reader: Callable,
        patch_size: Union[List[int], Tuple[int]],
        patch_generator: Optional[Callable],
        image_level_transform: Optional[Callable] = None,
        patch_level_transform: Optional[Callable] = None,
    ) -> None:
        """
        Parameters
        ----------
        data_path : str
            Path to data, must be a directory.
        data_reader : Callable
            Function that reads the image data from the file. Returns an iterable of image filenames.
        patch_size : Tuple[int]
            The size of the patch to extract from the image. Must be a tuple of len either 2 or 3 depending on number of spatial dimension in the data.
        patch_generator : Union[np.ndarray, Callable]
            Function that converts an input image (item from dataset) into a iterable of image patches.
            `patch_iter(dataset[idx])` must yield a tuple: (patches, coordinates).
        image_level_transform : Optional[Callable], optional
            _description_, by default None
        patch_level_transform : Optional[Callable], optional
            _description_, by default None
        """
        # #TODO Assert should all be done in Configuration.validate_wordir. Check

        self.data_path = data_path
        self.ext = ext
        self.axes = axes
        self.num_files = num_files
        self.data_reader = data_reader
        self.patch_size = patch_size
        self.patch_generator = patch_generator
        self.add_channel = patch_generator is not None
        self.image_transform = image_level_transform
        self.patch_transform = patch_level_transform

    @staticmethod
    def read_tiff_source(
        data_source: Union[str, Path], axes: str, patch_size: Tuple[int]
    ):
        """
        Read data source and correct dimensions.

        Parameters
        ----------
        data_source : str
            Path to data source

        add_channel : bool
            If True, add channel dimension to data source

        Returns
        -------
        image volume : np.ndarray
        """
        if not Path(data_source).exists():
            raise ValueError(f"Data source {data_source} does not exist")

        if data_source.suffix == ".npy":
            try:
                arr = np.load(data_source)
                arr_num_dims = len(arr.shape)
            except ValueError:
                arr = np.load(data_source, allow_pickle=True)
                arr_num_dims = (
                    len(arr[0].shape) + 1
                )  # TODO check all arrays have the same or compliant shape ?
        elif data_source.suffix[:4] == ".tif":
            arr = tifffile.imread(data_source)
            arr_num_dims = len(arr.shape)

        # remove any singleton dimensions
        arr = arr.squeeze()

        # sanity check on dimensions
        if len(arr.shape) < 2 or len(arr.shape) > 4:
            raise ValueError(
                f"Incorrect data dimensions. Must be 2, 3 or 4 (got {arr.shape} for file {data_source})."
            )

        # sanity check on axes length
        if len(axes) != len(arr.shape):
            raise ValueError(
                f"Incorrect axes length (got {axes} for file {data_source})."
            )

        # check axes validity
        are_axes_valid(axes)  # this raises errors

        # patch sanity check
        if len(patch_size) != len(arr.shape) and len(patch_size) != len(arr.shape) - 1:
            raise ValueError(
                f"Incorrect patch size (got {patch_size} for file {data_source} with shape {arr.shape})."
            )

        for p in patch_size:
            # check if power of 2
            if not (p & (p - 1) == 0):
                raise ValueError(
                    f"Incorrect patch size, should be power of 2 (got {patch_size} for file {data_source})."
                )

        # TODO add axes shuffling and reshapes. so far assuming correct order
        if ("S" in axes or "T" in axes) and arr.dtype != "O":
            arr = arr.reshape(
                -1, *arr.shape[len(axes.replace("Z", "").replace("YX", "")) :]
            )
        elif arr.dtype == "O":
            for i in range(len(arr)):
                arr[i] = np.expand_dims(arr[i], axis=0)
        else:
            arr = np.expand_dims(arr, axis=0)
        return arr

    def calculate_stats(self):
        mean = 0
        std = 0
        for i, image in tqdm(enumerate(self.__iter_source__())):
            mean += image.mean()
            std += np.std(image)
        self.mean = mean / (i + 1)
        self.std = std / (i + 1)
        logger.info(f"Calculated mean and std for {i + 1} images")
        logger.info(f"Mean: {self.mean}, std: {self.std}")

    def set_normalization(self, mean, std):
        self.mean = mean
        self.std = std

    def __len__(self):
        return self.num_files if self.num_files else len(self.source)

    def __iter_source_zarr__(self):
        # TODO better name?
        # load one zarr storage with zarr.open. Storage vs array? Check how it works with zarr
        # whether to read one bigger chunk and then patch or read patch by patch directly? compare speed
        # if it's zarr object type than read sample by sample. else read no less than 1 batch size ?
        # if chunk is in metadata, read no less than chunk size
        # Normalization? on Chunk or running?
        # reshape to 1d and then to user provided shape? test

        info = torch.utils.data.get_worker_info()
        num_workers = info.num_workers if info is not None else 1
        id = info.id if info is not None else 0

        # TODO add check if source is a valid zarr object
        self.source = zarr.open(Path(self.data_path), mode="r")

        if isinstance(self.source, zarr.core.Array):
            # TODO check if this is the correct way to get the shape
            self.source_shape = self.source.shape
            self.source = self.source.reshape(-1, *self.source_shape[1:])
            # TODO add checking chunk size ?

            # array should be of shape (S, (C), (Z), Y, X), iterating over S ?
            # TODO what if array is not of that shape and/or chunks aren't defined and
            if self.source.dtype == "O":
                # each sample is an array. Need patching in this case.
                for sample in range(self.source_shape[0]):
                    # start iterating over the source
                    # read chunk, reshape. #TODO this might be ok for random patching
                    if sample % num_workers == id:
                        yield self.image_transform(
                            self.source[sample]
                        ) if self.image_transform is not None else self.source[sample]
            else:
                # TODO add support for reshaping arbitraty number of dimensions
                # start iterating over the source
                # read chunk, reshape. #TODO this might be ok for random patching or if
                # array is of shape (S, (C), (Z), Y, X), iterating over S
                num_samples = 0  # TODO how to define number of samples in this case?
                for sample in range(num_samples):
                    if sample % num_workers == id:
                        yield self.image_transform(
                            self.source[sample]
                        ) if self.image_transform is not None else self.source[sample]

        elif isinstance(self.source, zarr.hierarchy.Group):
            # TODO add support for groups
            pass

        elif isinstance(self.source, zarr.storage.DirectoryStore):
            # TODO add support for different types of storages
            pass

        else:
            raise ValueError(f"Unsupported zarr object type {type(self.source)}")

    def __iter_source_tiff__(self):
        """
        Iterate over data source and yield whole image. Optional transform is applied to the images.

        Yields
        ------
        np.ndarray
        """
        info = torch.utils.data.get_worker_info()
        num_workers = info.num_workers if info is not None else 1
        id = info.id if info is not None else 0
        self.source = (
            itertools.islice(
                Path(self.data_path).rglob(f"*.{self.ext}*"), self.num_files
            )
            if self.num_files
            else Path(self.data_path).rglob(f"*.{self.ext}*")
        )

        # TODO check for mem leaks, explicitly gc the arr after iterator is exhausted
        for i, filename in enumerate(self.source):
            try:
                # TODO add buffer, several images up to some memory limit?
                arr = self.read_tiff_source(filename, self.axes, self.patch_size)
            except (ValueError, FileNotFoundError, OSError) as e:
                logging.exception(f"Exception in file {filename}, skipping")
                raise e
            if i % num_workers == id:
                # TODO add iterator inside
                yield self.image_transform(
                    arr
                ) if self.image_transform is not None else arr

    def __iter_source__(self):
        return (
            self.__iter_source_zarr__()
            if self.ext == "zarr"
            else self.__iter_source_tiff__()
        )

    def __iter__(self):
        """
        Iterate over data source and yield single patch. Optional transform is applied to the patches.

        Yields
        ------
        np.ndarray
        """

        for image in self.__iter_source__():
            if self.patch_generator is None:
                for idx in range(image.shape[0]):
                    sample = np.expand_dims(image[idx], (0, 1)).astype(
                        np.float32
                    )  # TODO check explanddims !!
                    yield normalize(sample, self.mean, self.std) if (
                        self.mean and self.std
                    ) else image
            else:
                for patch_data in self.patch_generator(
                    image, self.patch_size, mean=self.mean, std=self.std
                ):
                    yield self.patch_transform(
                        patch_data
                    ) if self.patch_transform is not None else (patch_data)
