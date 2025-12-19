from collections.abc import Sequence
from pathlib import Path
from typing import Any, Generic, Literal, NamedTuple, Union

import numpy as np
from numpy.typing import NDArray
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from careamics.config.data.ng_data_config import Mode, NGDataConfig, WholePatchingConfig
from careamics.config.transformations import NormalizeConfig
from careamics.dataset.dataset_utils.running_stats import WelfordStatistics
from careamics.dataset.patching.patching import Stats
from careamics.transforms import Compose

from .image_stack import GenericImageStack, ZarrImageStack
from .patch_extractor import PatchExtractor
from .patch_filter import create_coord_filter, create_patch_filter
from .patching_strategies import (
    PatchSpecs,
    RegionSpecs,
    create_patching_strategy,
)


class ImageRegionData(NamedTuple, Generic[RegionSpecs]):
    """
    Data structure for arrays produced by the dataset and propagated through models.

    An ImageRegionData may be a patch during training/validation, a tile during
    prediction with tiling, or a whole image during prediction without tiling.

    `data_shape` may not correspond to the shape of the original data if a subset
    of the channels has been requested, in which case the channel dimension may
    be smaller than that of the original data and only correspond to the requested
    number of channels.

    ImageRegionData may be collated in batches during training by the DataLoader. In
    that case:
    - data: arrays are collated into NDArray of shape (B,C,Z,Y,X)
    - source: list of str, length B
    - data_shape: list of tuples of int, each tuple being of length B and representing
        the shape of the original images in the corresponding dimension
    - dtype: list of str, length B
    - axes: list of str, length B
    - region_spec: dict of {str: sequence}, each sequence being of length B
    - additional_metadata: list of dict

    Description of the fields is given for the uncollated case (non-batched).
    """

    data: NDArray
    """Patch, tile or image in C(Z)YX format."""

    source: Union[str, Literal["array"]]
    """Source of the data, e.g. file path, zarr URI, or "array" for in-memory arrays."""

    data_shape: Sequence[int]
    """Shape of the original image in (SCZ)YX format and order. If channels are
    subsetted, the channel dimension corresponds to the number of requested channels."""

    dtype: str  # dtype should be str for collate
    """Data type of the original image as a string."""

    axes: str
    """Axes of the original data array, in format SCZYX."""

    region_spec: RegionSpecs  # PatchSpecs or subclasses, e.g. TileSpecs
    """Specifications of the region within the original image from where `data` is
    extracted. Of type PatchSpecs during training/validation and prediction without
    tiling, and TileSpecs during prediction with tiling.
    """

    additional_metadata: dict[str, Any]
    """Additional metadata to be stored with the image region. Currently used to store
    chunk and shard information for zarr image stacks."""


InputType = Union[Sequence[NDArray[Any]], Sequence[Path]]


def _adjust_shape_for_channels(
    shape: Sequence[int],
    channels: Sequence[int] | None,
    value: int | Literal["channels"] = "channels",
) -> tuple[int, ...]:
    """Adjust shape to account for channel subsetting.

    Parameters
    ----------
    shape : Sequence[int]
        The original data shape in SC(Z)YX format.
    channels : Sequence[int] | None
        The list of channels to select. If None, no adjustment is made.
    value : int | Literal["channels"], default="channels"
        The value to replace the channel dimension with. If "channels", the length
        of the channels list is used, by default "channels".

    Returns
    -------
    tuple[int, ...]
        The adjusted data shape in SC(Z)YX format.
    """
    if channels is not None:
        adjusted_shape = list(shape)
        adjusted_shape[1] = len(channels) if value == "channels" else value
        return tuple(adjusted_shape)
    return tuple(shape)


def _patch_size_within_data_shapes(
    data_shapes: Sequence[Sequence[int]], patch_size: Sequence[int]
) -> bool:
    """Determine whether all the data_shapes are greater than the patch size.

    Parameters
    ----------
    data_shapes : Sequence[Sequence[int]]
        A sequence of data shapes. They must be in the format SC(Z)YX.
    patch_size : Sequence[int]
        A patch size that must specify the size of the patch in all the spatial
        dimensions, in the format (Z)YX.

    Returns
    -------
    bool
        If all the data shapes are greater than the patch size.
    """
    smaller_than_shapes = [
        # skip sample and channel dimension in data_shape
        (np.array(patch_size) < np.array(data_shape[2:])).all()
        for data_shape in data_shapes
    ]
    return all(smaller_than_shapes)


class CareamicsDataset(Dataset, Generic[GenericImageStack]):
    def __init__(
        self,
        data_config: NGDataConfig,
        input_extractor: PatchExtractor[GenericImageStack],
        target_extractor: PatchExtractor[GenericImageStack] | None = None,
        mask_extractor: PatchExtractor[GenericImageStack] | None = None,
    ) -> None:

        # Make sure all the image sizes are greater than the patch size for training
        data_shapes = [
            image_stack.data_shape for image_stack in input_extractor.image_stacks
        ]
        if data_config.mode != Mode.PREDICTING:
            if not isinstance(
                data_config.patching, WholePatchingConfig
            ) and not _patch_size_within_data_shapes(
                data_shapes, data_config.patching.patch_size
            ):
                raise ValueError(
                    "Not all images sizes are greater than the patch size for training "
                    "and validation."
                )

        self.config = data_config

        self.input_extractor = input_extractor
        self.target_extractor = target_extractor

        self.patch_filter = (
            create_patch_filter(self.config.patch_filter)
            if self.config.patch_filter is not None
            else None
        )
        self.coord_filter = (
            create_coord_filter(self.config.coord_filter, mask=mask_extractor)
            if self.config.coord_filter is not None and mask_extractor is not None
            else None
        )
        self.patch_filter_patience = self.config.patch_filter_patience

        self.patching_strategy = create_patching_strategy(
            data_shapes=self.input_extractor.shapes,
            patching_config=self.config.patching,
        )

        self.input_stats, self.target_stats = self._initialize_statistics()

        self.transforms = self._initialize_transforms()

    def _initialize_transforms(self) -> Compose | None:
        normalize = NormalizeConfig(
            image_means=self.input_stats.means,
            image_stds=self.input_stats.stds,
            target_means=self.target_stats.means,
            target_stds=self.target_stats.stds,
        )
        if self.config.mode == Mode.TRAINING:
            # TODO: initialize normalization separately depending on configuration
            return Compose(transform_list=[normalize] + list(self.config.transforms))

        # TODO: add TTA
        return Compose(transform_list=[normalize])

    def _calculate_stats(
        self, data_extractor: PatchExtractor[GenericImageStack]
    ) -> Stats:
        image_stats = WelfordStatistics()
        n_patches = self.patching_strategy.n_patches

        for idx in tqdm(range(n_patches), desc="Computing statistics"):
            patch_spec = self.patching_strategy.get_patch_spec(idx)
            patch = data_extractor.extract_channel_patch(
                data_idx=patch_spec["data_idx"],
                sample_idx=patch_spec["sample_idx"],
                channels=self.config.channels,
                coords=patch_spec["coords"],
                patch_size=patch_spec["patch_size"],
            )
            # TODO: statistics accept SCYX format, while patch is CYX
            image_stats.update(patch[None, ...], sample_idx=idx)

        image_means, image_stds = image_stats.finalize()
        return Stats(image_means, image_stds)

    # TODO: add running stats
    def _initialize_statistics(self) -> tuple[Stats, Stats]:
        if self.config.image_means is not None and self.config.image_stds is not None:
            input_stats = Stats(self.config.image_means, self.config.image_stds)
        else:
            input_stats = self._calculate_stats(self.input_extractor)

        target_stats = Stats((), ())

        if self.config.target_means is not None and self.config.target_stds is not None:
            target_stats = Stats(self.config.target_means, self.config.target_stds)
        elif self.target_extractor is not None:
            target_stats = self._calculate_stats(self.target_extractor)

        return input_stats, target_stats

    def __len__(self):
        return self.patching_strategy.n_patches

    def _create_image_region(
        self, patch: np.ndarray, patch_spec: PatchSpecs, extractor: PatchExtractor
    ) -> ImageRegionData:
        data_idx = patch_spec["data_idx"]
        image_stack: GenericImageStack = extractor.image_stacks[data_idx]

        # adjust the number of channels in data_shape if needed
        data_shape = _adjust_shape_for_channels(
            shape=image_stack.data_shape,
            channels=self.config.channels,
        )

        # additional metadata for zarr image stacks
        if isinstance(image_stack, ZarrImageStack):
            additional_metadata = {
                "chunks": image_stack.chunks,
            }

            if image_stack.shards is not None:
                additional_metadata["shards"] = image_stack.shards
        else:
            additional_metadata = {}

        return ImageRegionData(
            data=patch,
            source=str(image_stack.source),
            dtype=str(image_stack.data_dtype),
            data_shape=data_shape,
            # TODO: should it be axes of the original image instead?
            axes=self.config.axes,
            region_spec=patch_spec,
            additional_metadata=additional_metadata,
        )

    def _extract_patches(
        self, patch_spec: PatchSpecs
    ) -> tuple[NDArray, NDArray | None]:
        """Extract input and target patches based on patch specifications."""
        input_patch = self.input_extractor.extract_channel_patch(
            data_idx=patch_spec["data_idx"],
            sample_idx=patch_spec["sample_idx"],
            channels=self.config.channels,
            coords=patch_spec["coords"],
            patch_size=patch_spec["patch_size"],
        )

        target_patch = (
            self.target_extractor.extract_channel_patch(
                data_idx=patch_spec["data_idx"],
                sample_idx=patch_spec["sample_idx"],
                # TODO does not allow selecting different channels for target
                channels=self.config.channels,
                coords=patch_spec["coords"],
                patch_size=patch_spec["patch_size"],
            )
            if self.target_extractor is not None
            else None
        )
        return input_patch, target_patch

    def _get_filtered_patch(
        self, index: int
    ) -> tuple[NDArray[Any], NDArray[Any] | None, PatchSpecs]:
        """Extract a patch that passes filtering criteria with retry logic."""
        should_filter = self.config.mode == Mode.TRAINING and (
            self.patch_filter is not None or self.coord_filter is not None
        )
        empty_patch = True
        patch_filter_patience = self.patch_filter_patience  # reset patience

        while empty_patch and patch_filter_patience > 0:
            # query patches
            patch_spec = self.patching_strategy.get_patch_spec(index)

            # filter patch based on coordinates if needed
            if should_filter and self.coord_filter is not None:
                if self.coord_filter.filter_out(patch_spec):
                    patch_filter_patience -= 1

                    # TODO should we raise an error rather than silently accept patches?
                    # if patience runs out without ever finding coordinates
                    # then we need to guard against an exist before defining
                    # input_patch and target_patch
                    if patch_filter_patience != 0:
                        continue

            input_patch, target_patch = self._extract_patches(patch_spec)

            # filter patch based on values if needed
            if should_filter and self.patch_filter is not None:
                empty_patch = self.patch_filter.filter_out(input_patch)
                patch_filter_patience -= 1  # decrease patience
            else:
                empty_patch = False

        return input_patch, target_patch, patch_spec

    def __getitem__(
        self, index: int
    ) -> Union[tuple[ImageRegionData], tuple[ImageRegionData, ImageRegionData]]:
        input_patch, target_patch, patch_spec = self._get_filtered_patch(index)

        # apply transforms
        if self.transforms is not None:
            if self.target_extractor is not None:
                input_patch, target_patch = self.transforms(input_patch, target_patch)
            else:
                # TODO: compose doesn't return None for target patch anymore
                #   so have to do this annoying if else
                (input_patch,) = self.transforms(input_patch, target_patch)
                target_patch = None

        input_data = self._create_image_region(
            patch=input_patch, patch_spec=patch_spec, extractor=self.input_extractor
        )

        if target_patch is not None and self.target_extractor is not None:
            target_data = self._create_image_region(
                patch=target_patch,
                patch_spec=patch_spec,
                extractor=self.target_extractor,
            )
            return input_data, target_data
        else:
            return (input_data,)
