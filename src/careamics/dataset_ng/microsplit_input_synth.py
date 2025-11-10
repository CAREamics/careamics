from collections.abc import Callable, Sequence
from typing import Any, Literal, NamedTuple

import numpy as np
from numpy.typing import NDArray

from .patch_extractor import PatchExtractor
from .patch_extractor.image_stack import ImageStack
from .patch_filter import PatchFilterProtocol
from .patching_strategies import PatchingStrategy, PatchSpecs


# TODO: better name
# mirrors format of ImageRegionData
class UncorrelatedRegionData(NamedTuple):
    data: NDArray
    source: Sequence[str | Literal["array"]]
    data_shape: Sequence[Sequence[int]]
    dtype: Sequence[str]  # dtype should be str for collate
    # axes: Sequence[str]
    region_spec: Sequence[PatchSpecs]


def is_empty(filter: PatchFilterProtocol) -> Callable[[NDArray[Any]], bool]:
    def is_empty_check(patch: NDArray[Any]) -> bool:
        return filter.filter_out(patch)

    return is_empty_check


def is_not_empty(filter: PatchFilterProtocol) -> Callable[[NDArray[Any]], bool]:
    def is_not_empty_check(patch: NDArray[Any]) -> bool:
        return not filter.filter_out(patch)

    return is_not_empty_check


def create_microsplit_patch(
    patch_extractor: PatchExtractor[ImageStack], patch_specs: list[PatchSpecs]
) -> NDArray[Any]:
    patches = np.concat(
        [
            patch_extractor.extract_channel_patch(
                data_idx=patch_spec["data_idx"],
                sample_idx=patch_spec["sample_idx"],
                channel_idx=c,
                coords=patch_spec["coords"],
                patch_size=patch_spec["patch_size"],
            )
            for c, patch_spec in enumerate(patch_specs)
        ],
        axis=0,
    )
    # Add L dimension if not present
    # TODO: make this more robust: maybe PatchExtractor should have an ndim attribute
    n_spatial_dims = len(patch_specs[0]["patch_size"])
    lateral_context_present = len(patches.shape) - n_spatial_dims == 2
    if not lateral_context_present:
        # insert a L dim
        patches = patches[:, np.newaxis]

    return patches


def get_random_channel_patches(
    idx: int,
    patching_strategy: PatchingStrategy,
    patch_extractor: PatchExtractor[ImageStack],
    rng: np.random.Generator | None,
) -> tuple[NDArray[Any], list[PatchSpecs]]:
    if rng is None:
        rng = np.random.default_rng()

    n_channels = patch_extractor.n_channels

    # in the original dataset, new random indices are chosen for each channel
    # the other channels can come from anywhere in the entire dataset
    indices = (idx, *rng.integers(patching_strategy.n_patches, size=(n_channels - 1)))

    # get n different patch specs for n different channels
    patch_specs = [patching_strategy.get_patch_spec(i) for i in indices]
    patches = create_microsplit_patch(patch_extractor, patch_specs)

    return patches, patch_specs


def get_empty_channel_patches(
    idx: int,
    patching_strategy: PatchingStrategy,
    patch_extractor: PatchExtractor,
    filled_channels: dict[int, PatchFilterProtocol],
    empty_channels: dict[int, PatchFilterProtocol],
    patience: int,
    rng: np.random.Generator | None,
) -> tuple[NDArray[Any], list[PatchSpecs]]:
    if rng is None:
        rng = np.random.default_rng()

    # if a channel is not selected to be empty or filled it will from idx
    filled = set(filled_channels.keys())
    empty = set(empty_channels.keys())
    if len(intersect := filled.intersection(empty)) != 0:
        raise ValueError(
            "Channels cannot be selected as both empty and filled, the following "
            f"channels were selected as both {intersect}."
        )

    n_channels = patch_extractor.n_channels
    patch_spec = patching_strategy.get_patch_spec(idx)
    patch_specs = [patch_spec for _ in range(n_channels)]
    # initial patches
    patches = create_microsplit_patch(patch_extractor, patch_specs)  # CL(Z)YX

    # for each channel sample patches until they are empty or not empty
    for c in range(n_channels):

        # criterion for the while loop
        criterion: Callable[[NDArray[Any]], bool]
        filter_: PatchFilterProtocol
        if c in empty_channels:
            filter_ = empty_channels[c]
            criterion = is_not_empty(filter_)
        elif c in filled_channels:
            filter_ = filled_channels[c]
            criterion = is_empty(filter_)
        else:
            break

        patch = patches[c]
        patience_ = patience
        # only check if primary input is empty
        while criterion(patch[0]) and patience_ > 0:
            # sample random indices from anywhere in the dataset
            new_idx = rng.integers(patching_strategy.n_patches)
            patch_spec = patching_strategy.get_patch_spec(new_idx.item())
            patch = patch_extractor.extract_channel_patch(
                data_idx=patch_spec["data_idx"],
                sample_idx=patch_spec["sample_idx"],
                channel_idx=c,
                coords=patch_spec["coords"],
                patch_size=patch_spec["patch_size"],
            )[0]
            # ^ removing channel dim
            patience_ -= 1
        if patience <= 0:
            print(f"Out of patience finding patch for channel {c}")

        patches[c] = patch
        patch_specs[c] = patch_spec

    return patches, patch_specs


def create_microsplit_input_target(
    patches: NDArray[Any],
    patch_specs: list[PatchSpecs],
    alphas: list[float],
    patch_extractor: PatchExtractor[ImageStack],
):
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
    )
    target_region = UncorrelatedRegionData(
        data=target_patch,
        source=source,
        data_shape=data_shape,
        dtype=dtype,
        region_spec=patch_specs,
    )
    return input_region, target_region
