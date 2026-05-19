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
from careamics.dataset.augmentation.compose import Compose
from careamics.models.constraints import ModelConstraints

from .image_region_data import ImageRegionData
from .image_stack import GenericImageStack
from .normalization import create_normalization
from .normalization.statistics import resolve_normalization_config
from .patch_constructor import PatchConstructor
from .patching import PatchSpecs, RegionSpecs

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


def _adjust_original_shape_for_channels(
    original_data_shape: Sequence[int],
    channels: Sequence[int] | None,
    axes: str,
    value: int | Literal["channels"] = "channels",
) -> Sequence[int]:
    # adjust original_data_shape for channel subsetting if needed
    if channels is not None and "C" in axes:
        c_idx = axes.index("C")
        adjusted_original_shape = list(original_data_shape)
        adjusted_original_shape[c_idx] = len(channels) if value == "channels" else value
        original_data_shape = tuple(adjusted_original_shape)
    return original_data_shape


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
    patch_arr = np.array(patch_size)
    return all(
        (patch_arr <= np.array(data_shape[2:])).all() for data_shape in data_shapes
    )


def _shapes_all_equal(data_shapes: Sequence[Sequence[int]]) -> bool:
    """Determine whether all the data shapes are equal.

    Parameters
    ----------
    data_shapes : Sequence[Sequence[int]]
        A sequence of data shapes. They must be in the format SC(Z)YX.

    Returns
    -------
    bool
        If all the data shapes are equal.
    """
    if not data_shapes:
        return True
    return all(shape == data_shapes[0] for shape in data_shapes[1:])


def _validate_shapes_against_model(
    data_config: DataConfig,
    model_constraints: ModelConstraints,
    data_shapes: Sequence[Sequence[int]],
    target_data_shapes: Sequence[Sequence[int]] | None = None,
) -> None:
    """Validate that the data shapes are compatible with the model constraints.

    Since patch and tile sizes are already validated in the configuration, this function
    validates that for whole patching all the data shapes are compatible with the model
    constraints.

    Finally, it validates the channels if no channel subset is specified.

    Parameters
    ----------
    data_config : DataConfig
        Dataset configuration.
    model_constraints : ModelConstraints
        The model constraints to validate against.
    data_shapes : Sequence[Sequence[int]]
        A sequence of data shapes. They must be in the format SC(Z)YX.
    target_data_shapes : Sequence[Sequence[int]] | None, default=None
        A sequence of target data shapes. They must be in the format SC(Z)YX.

    Raises
    ------
    ValueError
        If any of the data shapes is not compatible with the model constraints.
    """
    # model constraints not applied if there is a patch size (tiling, patching),
    # since it is validated in the configuration
    if isinstance(data_config.patching, WholePatchingConfig):
        for shape in data_shapes:
            model_constraints.validate_spatial_shape(shape[2:])

    # validate channels if not using a subset of channels, note that `channels`
    # is already validated against the model in the configuration
    if data_config.channels is None:
        # validate input channels
        for shape in data_shapes:
            model_constraints.validate_input_channels(shape[1])

        # validate target channels
        if target_data_shapes:
            for shape in target_data_shapes:
                model_constraints.validate_target_channels(shape[1])


def _validate_shapes_against_mode(
    data_config: DataConfig,
    data_shapes: Sequence[Sequence[int]],
) -> None:
    """Validate the input shapes against the mode and patching.

    Parameters
    ----------
    data_config : DataConfig
        Dataset configuration.
    data_shapes : Sequence[Sequence[int]]
        A sequence of data shapes. They must be in the format SC(Z)YX.

    Raises
    ------
    ValueError
        If the input shapes are not compatible with the mode and patching strategy.
    """
    # validate shapes according to the mode and patching strategy
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
    else:
        if isinstance(data_config.patching, WholePatchingConfig):
            if data_config.batch_size > 1 and not _shapes_all_equal(data_shapes):
                raise ValueError(
                    "For prediction without tiling, all images must have the same "
                    "size when batch size is greater than 1. Consider using a batch "
                    "size of 1 or use tiling."
                )


class CareamicsDataset(Dataset, Generic[GenericImageStack]):
    """PyTorch Dataset for CAREamics.

    Parameters
    ----------
    data_config : DataConfig
        Dataset configuration.
    patch_constructor : PatchConstructor
        Constructor for input and target patches.
    model_constraints : ModelConstraints, default=None
        If provided, the dataset will validate that the input patch size is compatible
        with the model constraints. Only used for prediction datasets.
    """

    def __init__(
        self,
        data_config: DataConfig,
        patch_constructor: PatchConstructor,
        model_constraints: ModelConstraints | None = None,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        data_config : DataConfig
            Dataset configuration.
        patch_constructor : PatchConstructor
            Constructor for input and target patches.
        model_constraints : ModelConstraints, default=None
            If provided, the dataset will validate that the input spatial shape is
            compatible with the model constraints.
        """
        # sanity checks on the input and output data
        input_shapes = patch_constructor.input_shapes
        target_shapes = patch_constructor.target_shapes
        _validate_shapes_against_mode(data_config, patch_constructor.input_shapes)

        if model_constraints is not None:
            _validate_shapes_against_model(
                data_config=data_config,
                model_constraints=model_constraints,
                data_shapes=input_shapes,
                target_data_shapes=target_shapes,
            )

        self.config = data_config
        self.patch_constructor = patch_constructor

        resolve_normalization_config(
            norm_config=self.config.normalization,
            patch_constructor=self.patch_constructor,
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

    def _create_image_regions(
        self,
        input_patch: NDArray[Any],
        target_patch: NDArray[Any] | None,
        patch_spec: RegionSpecs,
    ) -> tuple[ImageRegionData[RegionSpecs], ImageRegionData[RegionSpecs] | None]:
        """Wrap patches and their spec into ImageRegionData objects.

        Parameters
        ----------
        input_patch : NDArray
            Input patch data.
        target_patch : NDArray or None
            Optional target patch data.
        patch_spec : RegionSpecs
            Patch specification.

        Returns
        -------
        tuple of ImageRegionData and ImageRegionData or None
            Region data for the input patch and optional target patch.
        """
        data_idx = patch_spec["data_idx"]
        input_metadata = self.patch_constructor.get_input_image_metadata(data_idx)
        target_metadata = self.patch_constructor.get_target_image_metadata(data_idx)

        input_metadata["data_shape"] = _adjust_shape_for_channels(
            input_metadata["data_shape"], self.config.channels
        )
        input_metadata["original_data_shape"] = _adjust_original_shape_for_channels(
            input_metadata["original_data_shape"],
            self.config.channels,
            self.config.axes,
        )
        if target_metadata is not None:
            target_metadata["data_shape"] = _adjust_shape_for_channels(
                target_metadata["data_shape"], self.config.channels
            )
            target_metadata["original_data_shape"] = (
                _adjust_original_shape_for_channels(
                    target_metadata["original_data_shape"],
                    self.config.channels,
                    self.config.axes,
                )
            )

        input_data = ImageRegionData(
            data=input_patch,
            axes=self.config.axes,
            region_spec=patch_spec,
            **input_metadata,
        )
        if target_patch is not None and target_metadata is not None:
            target_data = ImageRegionData(
                data=target_patch,
                axes=self.config.axes,
                region_spec=patch_spec,
                **target_metadata,
            )
        else:
            target_data = None
        return input_data, target_data

    def __len__(self):
        """Return the number of patches (length of the dataset).

        Returns
        -------
        int
            Number of patches.
        """
        return self.patch_constructor.n_patches

    def __getitem__(
        self, index: int
    ) -> (
        tuple[ImageRegionData[PatchSpecs]]
        | tuple[ImageRegionData[PatchSpecs], ImageRegionData[PatchSpecs]]
    ):
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
        input_patch, target_patch, patch_spec = self.patch_constructor.construct_patch(
            index
        )

        # apply normalization
        input_patch, target_patch = self.normalization(input_patch, target_patch)

        # apply transforms
        if self.transforms is not None:
            if target_patch is not None:
                input_patch, target_patch = self.transforms(input_patch, target_patch)
            else:
                # TODO: compose doesn't return None for target patch anymore
                #   so have to do this annoying if else
                (input_patch,) = self.transforms(input_patch, target_patch)
                target_patch = None

        input_data, target_data = self._create_image_regions(
            input_patch, target_patch, patch_spec
        )
        if target_data is None:
            return (input_data,)
        else:
            return input_data, target_data
