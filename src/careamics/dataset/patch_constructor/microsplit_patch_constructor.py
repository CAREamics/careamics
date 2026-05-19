from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from skimage.transform import resize

from careamics.dataset.image_stack import ImageStack
from careamics.dataset.patch_extractor import PatchExtractor
from careamics.dataset.patching import (
    Patching,
    PatchSpecs,
    TiledPatching,
    TileSpecs,
    UncorrelatedPatchSpecs,
)

from .patch_constructor import PatchConstructor

# placeholder names


# target channels acquired together, synthetic sum input
class MsT1PatchConstructor(PatchConstructor):
    def __init__(
        self,
        patching_strategy: Patching,
        target_extractor: PatchExtractor[Any],
        multiscale_count: int,
        padding_mode: Literal["reflect", "wrap"],
        alpha_ranges: Sequence[tuple[float, float]],
        uncorrelated_channel_prob: float,
        channels: Sequence[int] | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.rng = rng if rng is not None else np.random.default_rng()

        self.patching_strategy = patching_strategy
        self.target_extractor = target_extractor
        self.channels = channels
        self.alpha_ranges = alpha_ranges
        self.multiscale_count = multiscale_count
        self.padding_mode: Literal["reflect", "wrap"] = padding_mode
        self.uncorrelated_channel_prob = uncorrelated_channel_prob

    @property
    def n_patches(self) -> int:
        return self.patching_strategy.n_patches

    def construct_patch(
        self, index: int
    ) -> tuple[NDArray[Any], NDArray[Any], PatchSpecs | UncorrelatedPatchSpecs]:
        p = self.rng.random()
        patch_spec: PatchSpecs | UncorrelatedPatchSpecs
        if p < self.uncorrelated_channel_prob:
            n_channels = (
                self.target_extractor.n_channels
                if self.channels is None
                else len(self.channels)
            )
            # TODO: randomly sampling indices will not work well with iterable files
            indices: list[int] = [index] + self.rng.integers(
                0, self.n_patches, n_channels - 1
            ).tolist()
            patch_specs = [self.patching_strategy.get_patch_spec(i) for i in indices]
            patch, patch_spec = _construct_uncorrelated_patch(
                self.target_extractor,
                patch_specs,
                channels=self.channels,
                principal_channel=0,
                multiscale_count=self.multiscale_count,
                padding_mode=self.padding_mode,
            )
        else:
            patch_spec = self.patching_strategy.get_patch_spec(index)
            patch = _extract_lc_patch(
                self.target_extractor,
                **patch_spec,
                channels=self.channels,
                multiscale_count=self.multiscale_count,
                padding_mode=self.padding_mode,
            )
        # patch has axes CL(Z)YX

        input_patch, target_patch = _create_input_target(
            patch, self.alpha_ranges, self.rng
        )
        return input_patch, target_patch, patch_spec

    def _construct_uncorrelated_patch(
        self, index: int
    ) -> tuple[NDArray[Any], UncorrelatedPatchSpecs]:
        indices: list[int] = [index] + self.rng.integers(
            0, self.n_patches, self.target_extractor.n_channels - 1
        ).tolist()
        patch_specs = [self.patching_strategy.get_patch_spec(i) for i in indices]

        n_channels = (
            self.target_extractor.n_channels
            if self.channels is None
            else len(self.channels)
        )
        patch_size = patch_specs[0]["patch_size"]
        patch = np.zeros((n_channels, self.multiscale_count, *patch_size))
        for c in range(n_channels):
            patch[[c]] = _extract_lc_patch(
                self.target_extractor,
                **patch_specs[c],
                channels=[c],
                multiscale_count=self.multiscale_count,
                padding_mode=self.padding_mode,
            )
        uncorr_patch_specs = UncorrelatedPatchSpecs(
            **patch_specs[0],
            principle_channel=0,
            all_data_idx=[ps["data_idx"] for ps in patch_specs],
            all_sample_idx=[ps["sample_idx"] for ps in patch_specs],
            all_coords=[ps["coords"] for ps in patch_specs],
        )
        return patch, uncorr_patch_specs

    def get_principal_input(self, input_patch: NDArray[Any]) -> NDArray[Any]:
        return input_patch[0]


# target channels in separate files, synthetic input
class MsT2PatchConstructor(PatchConstructor):
    def __init__(
        self,
        patching_strategies: Sequence[Patching],
        target_extractors: Sequence[PatchExtractor[Any]],
        multiscale_count: int,
        padding_mode: Literal["reflect", "wrap"],
        alpha_ranges: Sequence[tuple[float, float]],
        rng: np.random.Generator | None = None,
    ):
        self.rng = rng if rng is not None else np.random.default_rng()

        self.patching_strategies = patching_strategies
        self.target_extractors = target_extractors
        self.multiscale_count = multiscale_count
        self.alpha_ranges = alpha_ranges
        self.padding_mode: Literal["reflect", "wrap"] = padding_mode

        self.principal_channel = np.argmax(
            [patching.n_patches for patching in self.patching_strategies]
        ).item()

    @property
    def n_patches(self) -> int:
        return self.patching_strategies[self.principal_channel].n_patches

    def construct_patch(
        self, index: int
    ) -> tuple[NDArray[Any], NDArray[Any], UncorrelatedPatchSpecs]:
        n_channels = len(self.patching_strategies)
        n_patches = [
            patching_strategy.n_patches
            for patching_strategy in self.patching_strategies
        ]
        indices = self.rng.integers(0, n_patches, n_channels).tolist()
        indices[self.principal_channel] = index
        patch_specs = [
            strategy.get_patch_spec(i)
            for strategy, i in zip(self.patching_strategies, indices, strict=True)
        ]

        patch, uncorr_patch_specs = _construct_uncorrelated_patch(
            self.target_extractors,
            patch_specs,
            channels=None,
            principal_channel=self.principal_channel,
            multiscale_count=self.multiscale_count,
            padding_mode=self.padding_mode,
        )
        input_patch, target_patch = _create_input_target(
            patch, self.alpha_ranges, self.rng
        )
        return input_patch, target_patch, uncorr_patch_specs

    def get_principal_input(self, input_patch: NDArray[Any]) -> NDArray[Any]:
        return input_patch[0]


# real target image and input images
class MsT3PatchConstructor(PatchConstructor):
    def __init__(
        self,
        patching_strategy: Patching,
        input_extractor: PatchExtractor[Any],
        target_extractor: PatchExtractor[Any],
        multiscale_count: int,
        padding_mode: Literal["reflect", "wrap"],
    ):
        self.patching_strategy = patching_strategy
        self.input_extractor = input_extractor
        self.target_extractor = target_extractor
        self.multiscale_count = multiscale_count
        self.padding_mode: Literal["reflect", "wrap"] = padding_mode

    @property
    def n_patches(self) -> int:
        return self.patching_strategy.n_patches

    def construct_patch(
        self, index: int
    ) -> tuple[NDArray[Any], NDArray[Any], PatchSpecs]:
        patch_spec = self.patching_strategy.get_patch_spec(index)
        input_patch = self.input_extractor.extract_patch(**patch_spec)
        input_patch = _extract_lc_patch(
            self.input_extractor,
            **patch_spec,
            channels=None,
            multiscale_count=self.multiscale_count,
            padding_mode=self.padding_mode,
        )
        input_patch = input_patch.squeeze(0)  # output is CL(Z)YX
        target_patch = self.target_extractor.extract_patch(**patch_spec)
        return input_patch, target_patch, patch_spec

    def get_principal_input(self, input_patch: NDArray[Any]) -> NDArray[Any]:
        return input_patch[0]


class MsPredPatchConstructor(PatchConstructor):
    # prediction - input only
    def __init__(
        self,
        patching_strategy: TiledPatching,
        input_extractor: PatchExtractor[Any],
        multiscale_count: int,
        padding_mode: Literal["reflect", "wrap"],
    ):
        self.patching_strategy = patching_strategy
        self.input_extractor = input_extractor
        self.multiscale_count = multiscale_count
        self.padding_mode: Literal["reflect", "wrap"] = padding_mode

    @property
    def n_patches(self) -> int:
        return self.patching_strategy.n_patches

    def construct_patch(self, index: int) -> tuple[NDArray[Any], None, TileSpecs]:
        patch_spec = self.patching_strategy.get_patch_spec(index)
        input_patch = _extract_lc_patch(
            self.input_extractor,
            data_idx=patch_spec["data_idx"],
            sample_idx=patch_spec["sample_idx"],
            coords=patch_spec["coords"],
            patch_size=patch_spec["patch_size"],
            channels=None,
            multiscale_count=self.multiscale_count,
            padding_mode=self.padding_mode,
        )
        input_patch = input_patch.squeeze(0)  # output is CL(Z)YX
        return input_patch, None, patch_spec

    def get_principal_input(self, input_patch: NDArray[Any]) -> NDArray[Any]:
        return input_patch[0]


def _sample_alphas(
    alpha_ranges: Sequence[tuple[float, float]], rng: np.random.Generator
):
    low = tuple(r[0] for r in alpha_ranges)
    high = tuple(r[1] for r in alpha_ranges)
    return rng.uniform(low, high)


def _create_input_target(
    patch: NDArray[Any],
    alpha_ranges: Sequence[tuple[float, float]],
    rng: np.random.Generator,
):
    alphas = _sample_alphas(alpha_ranges, rng)
    alpha_broadcast = alphas.reshape(len(alphas), *(1,) * (len(patch.shape) - 1))
    # weight channels by alphas then sum on the channel axis
    # input dims will be L(Z)YX
    input_patch = (alpha_broadcast * patch).sum(axis=0)
    target_patch = patch[:, 0, ...]  # first L patch
    return input_patch, target_patch


def _construct_uncorrelated_patch(
    extractor: PatchExtractor[Any] | Sequence[PatchExtractor[Any]],
    patch_specs: Sequence[PatchSpecs],
    channels: Sequence[int] | None,
    principal_channel: int,
    multiscale_count: int,
    padding_mode: Literal["reflect", "wrap"] = "reflect",
) -> tuple[NDArray[Any], UncorrelatedPatchSpecs]:
    # TODO: cannot use channels with sequence of extractors

    n_channels = len(patch_specs)
    patch_size = patch_specs[0]["patch_size"]
    patch = np.zeros((n_channels, multiscale_count, *patch_size))
    for c_idx in range(n_channels):
        if isinstance(extractor, PatchExtractor):
            original_c_idx = c_idx if channels is None else channels[c_idx]
            patch[[c_idx]] = _extract_lc_patch(
                extractor,
                **patch_specs[c_idx],
                channels=[original_c_idx],
                multiscale_count=multiscale_count,
                padding_mode=padding_mode,
            )
        else:
            patch[[c_idx]] = _extract_lc_patch(
                extractor[c_idx],
                **patch_specs[c_idx],
                # TODO: guard that sequence of extractors each have n_channels==1
                channels=None,
                multiscale_count=multiscale_count,
                padding_mode=padding_mode,
            )
    uncorr_patch_specs = UncorrelatedPatchSpecs(
        **patch_specs[principal_channel],
        principle_channel=principal_channel,
        all_data_idx=[ps["data_idx"] for ps in patch_specs],
        all_sample_idx=[ps["sample_idx"] for ps in patch_specs],
        all_coords=[ps["coords"] for ps in patch_specs],
    )
    return patch, uncorr_patch_specs


def _extract_lc_patch(
    extractor: PatchExtractor[ImageStack],
    data_idx: int,
    sample_idx: int,
    channels: Sequence[int] | None,
    coords: Sequence[int],
    patch_size: Sequence[int],
    multiscale_count: int,
    padding_mode: Literal["reflect", "wrap"] = "reflect",
) -> NDArray[Any]:
    shape = extractor.image_stacks[data_idx].data_shape
    spatial_shape = shape[2:]
    n_channels = shape[1] if channels is None else len(channels)

    final_lc_patch_size = np.array(patch_size) * (2**multiscale_count)
    final_lc_start = (
        np.array(coords) + np.array(patch_size) // 2 - final_lc_patch_size // 2
    )
    final_lc_end = final_lc_start + np.array(final_lc_patch_size)

    start_clipped = np.clip(
        final_lc_start, np.zeros_like(spatial_shape), np.array(spatial_shape)
    )
    end_clipped = np.clip(
        final_lc_end, np.zeros_like(spatial_shape), np.array(spatial_shape)
    )
    size_clipped = end_clipped - start_clipped

    final_lc_patch = extractor.extract_channel_patch(
        data_idx, sample_idx, channels, start_clipped, size_clipped
    )
    pad_before = start_clipped - final_lc_start
    pad_after = final_lc_end - end_clipped
    pad_width = np.concatenate(
        [
            # zeros to not pad the channel axis
            np.zeros((1, 2), dtype=int),
            np.stack([pad_before, pad_after], axis=-1),
        ]
    )
    final_lc_patch = np.pad(
        final_lc_patch,
        pad_width,
        mode=padding_mode,
    )

    patch = np.zeros((n_channels, multiscale_count, *patch_size), dtype=np.float32)
    for scale in range(multiscale_count):
        lc_patch_size = np.array(patch_size) * (2**scale)
        lc_start = final_lc_patch_size // 2 - lc_patch_size // 2
        lc_end = lc_start + lc_patch_size

        spatial_slice = [slice(s, t) for s, t in zip(lc_start, lc_end, strict=True)]
        lc_patch = final_lc_patch[..., *spatial_slice]  # type: ignore
        # TODO: test different downscaling? skimage suggests downscale_local_mean
        lc_patch = resize(lc_patch, (n_channels, *patch_size))
        patch[:, scale, ...] = lc_patch
    return patch
