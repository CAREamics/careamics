"""CAREamics dataset and image region types."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Generic, Literal, Union

import numpy as np
from numpy.typing import NDArray
from torch.utils.data import Dataset

from careamics.config.data.data_config import (
    DataConfig,
    Mode,
    WholePatchingConfig,
)
from careamics.lightning.lightning_modules.constraints import (
    ModelConstraints,
)
from careamics.transforms.compose import Compose

from .image_region_data import ImageRegionData
from .image_stack import GenericImageStack, ZarrImageStack
from .normalization import create_normalization
from .normalization.statistics import resolve_normalization_config
from .patch_extractor import PatchExtractor
from .patching_strategies import (
    PatchingStrategy,
    PatchSpecs,
)

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
    """Determine whether all the data_shapes are greater or equal than the patch size.

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
        If all the data shapes are greater or equal than the patch size.
    """
    smaller_than_shapes = [
        # skip sample and channel dimension in data_shape
        (np.array(patch_size) <= np.array(data_shape[2:])).all()
        for data_shape in data_shapes
    ]
    return all(smaller_than_shapes)


class CareamicsDataset(Dataset, Generic[GenericImageStack]):
    """PyTorch Dataset for CAREamics.

    Parameters
    ----------
    data_config : DataConfig
        Dataset configuration.
    patching_strategy : PatchingStrategy
        Strategy for sampling patches.
    input_extractor : PatchExtractor
        Extractor for input patches.
    target_extractor : PatchExtractor or None, optional
        Extractor for target patches.
    model_constraints : ModelConstraints, optional
        If provided, the dataset will validate that the input patch size is compatible
        with the model constraints. Only used for prediction datasets.
    """

    def __init__(
        self,
        data_config: DataConfig,
        patching_strategy: PatchingStrategy,
        input_extractor: PatchExtractor[GenericImageStack],
        target_extractor: PatchExtractor[GenericImageStack] | None = None,
        model_constraints: ModelConstraints | None = None,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        data_config : DataConfig
            Dataset configuration.
        patching_strategy : PatchingStrategy
            Strategy for sampling patches.
        input_extractor : PatchExtractor
            Extractor for input patches.
        target_extractor : PatchExtractor or None, optional
            Extractor for target patches.
        model_constraints : ModelConstraints, optional
            If provided, the dataset will validate that the input spatial shape is
            compatible with the model constraints. Only used for prediction datasets.
        """
        # sanity checks on the input and output data
        data_shapes = [
            image_stack.data_shape for image_stack in input_extractor.image_stacks
        ]
        if data_config.mode != Mode.PREDICTING:
            # make sure all the image sizes are greater than the patch size for training
            if not isinstance(
                data_config.patching, WholePatchingConfig
            ) and not _patch_size_within_data_shapes(
                data_shapes, data_config.patching.patch_size
            ):
                raise ValueError(
                    "Not all images sizes are greater or equal than the patch size for "
                    "training and validation."
                )

        if model_constraints is not None:
            # model constraints not applied if there is a patch size (tiling, patching),
            # since it is validated in the configuration
            if isinstance(data_config.patching, WholePatchingConfig):
                for shape in data_shapes:
                    # raise errors if spatial shape is not compatible with model
                    # constraints
                    model_constraints.validate_spatial_shape(shape[2:])

            # validate channels if not using a subset of channels, note that `channels`
            # is already validated agaisnt the model in the configuration
            if data_config.channels is None:
                # validate input channels
                for shape in data_shapes:
                    model_constraints.validate_input_channels(shape[1])

                # validate target channels
                if target_extractor is not None:
                    for image_stack in target_extractor.image_stacks:
                        model_constraints.validate_target_channels(
                            image_stack.data_shape[1]
                        )

        self.config = data_config

        self.input_extractor = input_extractor
        self.target_extractor = target_extractor

        self.patching_strategy = patching_strategy

        resolve_normalization_config(
            norm_config=self.config.normalization,
            patching_strategy=self.patching_strategy,
            input_extractor=self.input_extractor,
            target_extractor=self.target_extractor,
            channels=self.config.channels,
        )
        self.normalization = create_normalization(self.config.normalization)

        self.transforms = self._initialize_augmentations()

    def _initialize_augmentations(self) -> Compose | None:
        """Build the composition of augmentations.

        Returns
        -------
        Compose or None
            Augmentations or empty compose.
        """
        if self.config.mode == Mode.TRAINING:
            return Compose(list(self.config.augmentations))

        # TODO: add TTA
        return Compose([])

    def __len__(self):
        """Return the number of patches (length of the dataset).

        Returns
        -------
        int
            Number of patches.
        """
        return self.patching_strategy.n_patches

    def _create_image_region(
        self, patch: np.ndarray, patch_spec: PatchSpecs, extractor: PatchExtractor
    ) -> ImageRegionData:
        """Wrap a patch and its spec into an ImageRegionData for the given extractor.

        Parameters
        ----------
        patch : np.ndarray
            Patch data.
        patch_spec : PatchSpecs
            Patch specification.
        extractor : PatchExtractor
            Extractor used (for metadata).

        Returns
        -------
        ImageRegionData
            Region data for the patch.
        """
        data_idx = patch_spec["data_idx"]
        image_stack: GenericImageStack = extractor.image_stacks[data_idx]

        # adjust the number of channels in data_shape if needed
        data_shape = _adjust_shape_for_channels(
            shape=image_stack.data_shape,
            channels=self.config.channels,
        )

        # get original shape from image stack
        original_data_shape = image_stack.original_data_shape

        # adjust original_data_shape for channel subsetting if needed
        if self.config.channels is not None and "C" in self.config.axes:
            c_idx = self.config.axes.index("C")
            adjusted_original_shape = list(original_data_shape)
            adjusted_original_shape[c_idx] = len(self.config.channels)
            original_data_shape = tuple(adjusted_original_shape)

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
            axes=self.config.axes,
            original_data_shape=original_data_shape,
            region_spec=patch_spec,
            additional_metadata=additional_metadata,
        )

    def _extract_patches(
        self, patch_spec: PatchSpecs
    ) -> tuple[NDArray, NDArray | None]:
        """Extract input and target patches based on patch specifications.

        Parameters
        ----------
        patch_spec : PatchSpecs
            Patch specification (data_idx, sample_idx, coords, patch_size).

        Returns
        -------
        tuple of (NDArray, NDArray or None)
            Input patch and optional target patch.
        """
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

    def __getitem__(
        self, index: int
    ) -> Union[tuple[ImageRegionData], tuple[ImageRegionData, ImageRegionData]]:
        """Return a tuple of ImageRegionData for the given index.

        Parameters
        ----------
        index : int
            Dataset index.

        Returns
        -------
        tuple of ImageRegionData
            (input_data,) or (input_data, target_data).
        """
        patch_spec = self.patching_strategy.get_patch_spec(index)
        input_patch, target_patch = self._extract_patches(patch_spec)

        # apply normalization
        input_patch, target_patch = self.normalization(input_patch, target_patch)

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
