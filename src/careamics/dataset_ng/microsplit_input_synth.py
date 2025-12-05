"""MicroSplit patch synthesis."""

# --- PROOF OF PRINCIPLE ---


from collections.abc import Callable, Sequence
from typing import Any, Literal, NamedTuple

import numpy as np
from numpy.typing import NDArray

from .dataset import ImageRegionData
from .image_stack import ImageStack
from .patch_extractor import PatchExtractor
from .patch_filter import PatchFilterProtocol
from .patching_strategies import PatchingStrategy, PatchSpecs


# TODO: better name
# mirrors format of ImageRegionData
class UncorrelatedRegionData(NamedTuple):
    data: NDArray
    source: Sequence[str | Literal["array"]]
    data_shape: Sequence[Sequence[int]]
    dtype: Sequence[str]  # dtype should be str for collate
    axes: Sequence[str]
    region_spec: Sequence[PatchSpecs]


# --- for finding empty / signal channel patches in loop
def is_empty(filter: PatchFilterProtocol) -> Callable[[NDArray[Any]], bool]:
    def is_empty_check(patch: NDArray[Any]) -> bool:
        return filter.filter_out(patch)

    return is_empty_check


def is_not_empty(filter: PatchFilterProtocol) -> Callable[[NDArray[Any]], bool]:
    def is_not_empty_check(patch: NDArray[Any]) -> bool:
        return not filter.filter_out(patch)

    return is_not_empty_check


# ---


def create_default_input_target(
    idx: int,
    patch_extractor: PatchExtractor[ImageStack],
    patching_strategy: PatchingStrategy,
    alphas: list[float],
    axes: str,  # annoyingly have to supply this to image region
) -> tuple[ImageRegionData, ImageRegionData]:
    """
    Create a default MicroSplit patch with synthetically summed input.

    Parameters
    ----------
    idx: int
        The dataset index.
    patch_extractor: PatchExtractor
        Used to extract patches from the data.
    patching_strategy: PatchingStrategy
        Patch locations will be sampled using the patching strategy.
    alphas: list[float]
        Weights for each channel for creating the synthetic input with summation.
    axes: str
        The axes of the data. This is only used to populate metadata.

    Returns
    -------
    input_region: ImageRegionData
        The input patch and its metadata, the data has the dimension L(Z)YX.
    target_region: ImageRegionData
        The target patch and its metadata, the data has the dimensions C(Z)YX.
    """
    patch_spec = patching_strategy.get_patch_spec(idx)
    patches = extract_microsplit_patch(patch_extractor, patch_spec)

    ndims = len(patches.shape) - 1
    alpha_broadcast = np.array(alphas)[:, *(np.newaxis for _ in range(ndims))]
    # weight channels by alphas then sum on the channel axis
    # input dims will be L(Z)YX
    input_patch = (alpha_broadcast * patches).sum(axis=0)
    target_patch = patches[:, 0, ...]  # first L patch

    data_idx = patch_spec["data_idx"]
    input_region = ImageRegionData(
        input_patch,
        source=str(patch_extractor.image_stacks[data_idx].source),
        data_shape=patch_extractor.image_stacks[data_idx].data_shape,
        dtype=str(patch_extractor.image_stacks[data_idx].data_dtype),
        axes=axes,
        region_spec=patch_spec,
        additional_metadata={},
    )
    target_region = ImageRegionData(
        target_patch,
        source=str(patch_extractor.image_stacks[data_idx].source),
        data_shape=patch_extractor.image_stacks[data_idx].data_shape,
        dtype=str(patch_extractor.image_stacks[data_idx].data_dtype),
        axes=axes,
        region_spec=patch_spec,
        additional_metadata={},
    )
    return input_region, target_region


def create_uncorrelated_input_target(
    patches: NDArray[Any],
    patch_specs: list[PatchSpecs],
    alphas: list[float],
    patch_extractor: PatchExtractor[ImageStack],  # for metadata
    axes: str,  # mirroring imageregion
) -> tuple[UncorrelatedRegionData, UncorrelatedRegionData]:
    """
    Create MicroSplit target and synthetically summed input with metadata.

    Parameters
    ----------
    patches: NDArray
        Patches with dimensions LC(Z)YX, where L contains the lateral context at
        multiple scales.
    patch_specs: list[PatchSpecs]
        The patch specs for each channel.
    alphas: list[float]
        Weights for each channel for creating the synthetic input with summation.
    patch_extractor: PatchExtractor
        The patch extractor the patches were extracted from. Used for additional
        metadata.

    Returns
    -------
    input_region: UncorrelatedRegionData
        The input patch and its metadata, the data has the dimension L(Z)YX.
    target_region: UncorrelatedRegionData
        The target patch and its metadata, the data has the dimensions C(Z)YX.
    """
    ndims = len(patches.shape) - 1
    alpha_broadcast = np.array(alphas)[:, *(np.newaxis for _ in range(ndims))]
    # weight channels by alphas then sum on the channel axis
    # input dims will be L(Z)YX
    input_patch = (alpha_broadcast * patches).sum(axis=0)
    target_patch = patches[:, 0, ...]  # first L patch

    input_stacks = [
        patch_extractor.image_stacks[patch_spec["data_idx"]]
        for patch_spec in patch_specs
    ]
    source = [str(stack.source) for stack in input_stacks]
    data_shape = [stack.data_shape for stack in input_stacks]
    dtype = [str(stack.data_dtype) for stack in input_stacks]

    input_region = UncorrelatedRegionData(
        data=input_patch,
        source=source,
        data_shape=data_shape,
        dtype=dtype,
        region_spec=patch_specs,
        axes=axes,
    )
    target_region = UncorrelatedRegionData(
        data=target_patch,
        source=source,
        data_shape=data_shape,
        dtype=dtype,
        region_spec=patch_specs,
        axes=axes,
    )
    return input_region, target_region


def get_random_channel_patches(
    idx: int,  # TODO: is this needed it makes it work the same as original dataset
    patch_extractor: PatchExtractor[ImageStack],
    patching_strategy: PatchingStrategy,
    rng: np.random.Generator | None,
) -> tuple[NDArray[Any], list[PatchSpecs]]:
    """
    Select patches form random patch locations for each channel.

    Parameters
    ----------
    idx: int
        The dataset index.
    patch_extractor: PatchExtractor
        Used to extract patches from the data.
    patching_strategy: PatchingStrategy
        Patch locations will be sampled using the patching strategy.
    rng: numpy.random.Generator | None
        Useful for seeding the process. If `None` the default random number generator
        will be used.

    Returns
    -------
    NDArray[Any]
        The resulting patches with dimensions LC(Z)YX, where L contains the lateral
        context at multiple scales.
    list[PatchSpecs]
        A list of patch specification, one for each channel.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_channels = patch_extractor.n_channels

    # in the original dataset, new random indices are chosen for each channel
    # the other channels can come from anywhere in the entire dataset
    indices = (idx, *rng.integers(patching_strategy.n_patches, size=(n_channels - 1)))

    # get n different patch specs for n different channels
    patch_specs = [patching_strategy.get_patch_spec(i) for i in indices]
    patches = extract_microsplit_patch(patch_extractor, patch_specs)

    return patches, patch_specs


# TODO: better name
def get_empty_channel_patches(
    idx: int,
    patch_extractor: PatchExtractor,
    patching_strategy: PatchingStrategy,
    signal_channels: dict[int, PatchFilterProtocol],
    empty_channels: dict[int, PatchFilterProtocol],
    patience: int,
    rng: np.random.Generator | None,
) -> tuple[NDArray[Any], list[PatchSpecs]]:
    """
    Select patches, specifying which channels should have signal and which should not.

    Parameters
    ----------
    idx: int
        The dataset index.
    patch_extractor: PatchExtractor
        Used to extract patches from the data.
    patching_strategy: PatchingStrategy
        Patch locations will be sampled using the patching strategy.
    signal_channels: dict[int, PatchFilterProtocol]
        A dictionary to specify the channels that should have signal and how they should
        be filtered. The keys are the channel index and the values are the patch filters
        used to determine if the channel patch is empty or not.
    empty_channels: dict[int, PatchFilterProtocol]
        A dictionary to specify the channels that should not have signal. Similar to
        the `signal_channels`.
    patience: int
        New patches are selected at random until a patch with signal or without is
        found, the `patience` determines how many times to look before giving up.
    rng: numpy.random.Generator | None
        Useful for seeding the process. If `None` the default random number generator
        will be used.

    Returns
    -------
    NDArray[Any]
        The resulting patches with dimensions LC(Z)YX, where L contains the lateral
        context at multiple scales.
    list[PatchSpecs]
        A list of patch specification, one for each channel.
    """
    if rng is None:
        rng = np.random.default_rng()

    # if a channel is not selected to be empty or filled it will from idx
    filled = set(signal_channels.keys())
    empty = set(empty_channels.keys())
    if len(intersect := filled.intersection(empty)) != 0:
        raise ValueError(
            "Channels cannot be selected as both empty and filled, the following "
            f"channels were selected as both {intersect}."
        )

    n_channels = patch_extractor.n_channels

    # start with random initial patches
    patches, patch_specs = get_random_channel_patches(
        idx, patch_extractor, patching_strategy, rng
    )

    # for each channel sample patches until they are empty or not empty
    for c in range(n_channels):

        # criterion for the while loop
        criterion: Callable[[NDArray[Any]], bool]
        filter_: PatchFilterProtocol
        if c in empty_channels:
            filter_ = empty_channels[c]
            criterion = is_not_empty(filter_)
        elif c in signal_channels:
            filter_ = signal_channels[c]
            criterion = is_empty(filter_)
        else:
            break

        patch = patches[c]
        patch_spec = patch_specs[c]
        patience_ = patience
        # only check if primary input is empty
        while criterion(patch[0]) and patience_ > 0:
            # sample random indices from anywhere in the dataset
            new_idx = rng.integers(patching_strategy.n_patches)
            patch_spec = patching_strategy.get_patch_spec(new_idx.item())
            patch = patch_extractor.extract_channel_patch(
                data_idx=patch_spec["data_idx"],
                sample_idx=patch_spec["sample_idx"],
                channels=[c],
                coords=patch_spec["coords"],
                patch_size=patch_spec["patch_size"],
            )[0]
            # ^ removing channel dim
            patience_ -= 1
        if patience <= 0:
            # TODO: log properly
            print(f"Out of patience finding patch for channel {c}")

        patches[c] = patch
        patch_specs[c] = patch_spec

    return patches, patch_specs


def extract_microsplit_patch(
    patch_extractor: PatchExtractor[ImageStack],
    patch_specs: PatchSpecs | list[PatchSpecs],
) -> NDArray[Any]:
    """
    Extract a MicroSplit patch with the dimensions LC(Z)YX.

    This patch can be used to synthesis an input patch by summing the C dimension, and
    it can be used to create a target patch by selecting the primary input from the
    L dimension, where L is to store lateral context patches.

    Parameters
    ----------
    patch_extractor: PatchExtractor
        Used to extract patches from the data.
    patch_specs: PatchSpec | list[PatchSpecs]
        A patch specification or a list of patch specifications — one for each channel.
        Different patch specs can be used or each channel to create uncorrelated channel
        patches.

    Returns
    -------
    NDArray[Any]
        The resulting patches with dimensions LC(Z)YX, where L contains the lateral
        context at multiple scales.
    """
    if isinstance(patch_specs, list):
        patches = np.concat(
            [
                patch_extractor.extract_channel_patch(
                    data_idx=patch_spec["data_idx"],
                    sample_idx=patch_spec["sample_idx"],
                    channels=[c],
                    coords=patch_spec["coords"],
                    patch_size=patch_spec["patch_size"],
                )
                for c, patch_spec in enumerate(patch_specs)
            ],
            axis=0,
        )
    else:
        patches = patch_extractor.extract_patch(
            data_idx=patch_specs["data_idx"],
            sample_idx=patch_specs["sample_idx"],
            coords=patch_specs["coords"],
            patch_size=patch_specs["patch_size"],
        )
    # Add L dimension if not present
    n_spatial_dims = patch_extractor.n_spatial_dims
    lateral_context_present = len(patches.shape) - n_spatial_dims == 2
    if not lateral_context_present:
        # insert a L dim
        patches = patches[:, np.newaxis]

    return patches
