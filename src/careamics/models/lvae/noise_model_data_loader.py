"""Data loading utilities for Gaussian mixture noise-model training."""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from careamics.config.support import SupportedData
from careamics.dataset.factory import (
    Loading,
    init_patch_extractor,
    select_image_stack_loader,
    select_patch_extractor_type,
)
from careamics.dataset.patch_extractor import PatchExtractor
from careamics.lightning.data.data_module_utils import initialize_data_pair

NoiseModelInput = (
    NDArray[Any] | Sequence[NDArray[Any]] | str | Path | Sequence[str | Path]
)


def create_noise_model_data_loader(
    signal: NoiseModelInput | Any,
    observation: NoiseModelInput | Any,
    *,
    axes: str,
    data_type: SupportedData | str,
    batch_size: int,
    n_epochs: int,
    loading: Loading = None,
    channel: int = 0,
    lower_clip: float = 0.0,
    upper_clip: float = 100.0,
) -> Iterator[NDArray[np.float32]]:
    """Yield batches of paired signal and observation intensities.

    When the available pixels fit within the training budget (`batch_size * n_epochs`),
    all pairs are loaded into memory. For larger data, aligned sequential patches are
    read and sampled evenly across signal-intensity deciles.

    Parameters
    ----------
    signal : array, path, or sequence of arrays or paths
        Clean signal data.
    observation : array, path, or sequence of arrays or paths
        Noisy observations corresponding to `signal`.
    axes : str
        Axes describing the signal and observation inputs.
    data_type : SupportedData or str
        CAREamics data type.
    batch_size : int
        Number of intensity pairs in each full batch.
    n_epochs : int
        Maximum number of batches to yield.
    loading : Loading, default=None
        Custom CAREamics loading strategy.
    channel : int, default=0
        Channel to extract from canonical `SC(Z)YX` data.
    lower_clip : float, default=0.0
        Lower signal percentile to retain.
    upper_clip : float, default=100.0
        Upper signal percentile to retain.

    Yields
    ------
    numpy.ndarray
        Float32 array of shape `(N, 2)` containing signal and observation
        intensities, where `N` is at most `batch_size`.
    """
    _validate_training_parameters(
        batch_size=batch_size,
        n_epochs=n_epochs,
        lower_clip=lower_clip,
        upper_clip=upper_clip,
    )

    signal_extractor, observation_extractor = _create_paired_extractors(
        signal=signal,
        observation=observation,
        axes=axes,
        data_type=SupportedData(data_type),
        loading=loading,
    )
    max_pixels = batch_size * n_epochs
    n_pixels = _number_of_pixels(signal_extractor.shapes)
    if n_pixels <= max_pixels:
        pairs = _load_all_pairs(
            signal_extractor, observation_extractor, channel=channel
        )
        yield from _yield_in_memory_batches(
            pairs,
            batch_size=batch_size,
            n_epochs=n_epochs,
            lower_clip=lower_clip,
            upper_clip=upper_clip,
        )
    else:
        yield from _yield_sampled_batches(
            signal_extractor,
            observation_extractor,
            channel=channel,
            batch_size=batch_size,
            n_epochs=n_epochs,
            lower_clip=lower_clip,
            upper_clip=upper_clip,
        )


def _validate_training_parameters(
    batch_size: int, n_epochs: int, lower_clip: float, upper_clip: float
) -> None:
    """Validate batching and percentile parameters.

    Parameters
    ----------
    batch_size : int
        Number of pairs per batch.
    n_epochs : int
        Maximum number of batches.
    lower_clip : float
        Lower clipping percentile.
    upper_clip : float
        Upper clipping percentile.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if n_epochs <= 0:
        raise ValueError(f"n_epochs must be positive, got {n_epochs}.")
    if not 0 <= lower_clip < upper_clip <= 100:
        raise ValueError(
            "Percentile bounds must satisfy "
            f"0 <= lower_clip < upper_clip <= 100, got {lower_clip} and "
            f"{upper_clip}."
        )


def _create_paired_extractors(
    signal: Any,
    observation: Any,
    axes: str,
    data_type: SupportedData,
    loading: Loading,
) -> tuple[PatchExtractor[Any], PatchExtractor[Any]]:
    """Initialize aligned CAREamics patch extractors.

    Parameters
    ----------
    signal : Any
        Signal data source.
    observation : Any
        Observation data source.
    axes : str
        Axes describing both sources.
    data_type : SupportedData
        CAREamics data type.
    loading : Loading
        Optional custom loading strategy.

    Returns
    -------
    tuple of PatchExtractor
        Signal and observation patch extractors.
    """
    signal_sources, observation_sources = initialize_data_pair(
        data_type=data_type,
        input_data=signal,
        target_data=observation,
        loading=loading,
    )
    assert observation_sources is not None

    # TIFF files can be loaded one at a time. Other built-in formats already use
    # in-memory or lazy ImageStack implementations.
    in_memory = data_type != SupportedData.TIFF
    extractor_type = select_patch_extractor_type(data_type, in_memory=in_memory)
    image_stack_loader = select_image_stack_loader(
        data_type=data_type, in_memory=in_memory, loading=loading
    )
    return (
        init_patch_extractor(extractor_type, image_stack_loader, signal_sources, axes),
        init_patch_extractor(
            extractor_type, image_stack_loader, observation_sources, axes
        ),
    )


def _number_of_pixels(shapes: Sequence[Sequence[int]]) -> int:
    """Return the number of pixels available to each channel.

    Parameters
    ----------
    shapes : sequence of sequences of int
        Canonical `SC(Z)YX` shapes.

    Returns
    -------
    int
        Total pixels per channel.
    """
    return sum(math.prod((shape[0], *shape[2:])) for shape in shapes)


def _load_all_pairs(
    signal_extractor: PatchExtractor[Any],
    observation_extractor: PatchExtractor[Any],
    *,
    channel: int,
) -> NDArray[np.float32]:
    """Load and combine every aligned intensity pair.

    Parameters
    ----------
    signal_extractor : PatchExtractor
        Signal patch extractor.
    observation_extractor : PatchExtractor
        Observation patch extractor.
    channel : int
        Channel selected for training.

    Returns
    -------
    numpy.ndarray
        All paired signal and observation intensities.
    """
    pair_parts: list[NDArray[np.float32]] = []
    for data_idx, shape in enumerate(signal_extractor.shapes):
        spatial_shape = tuple(shape[2:])
        coords = (0,) * len(spatial_shape)
        for sample_idx in range(shape[0]):
            signal = signal_extractor.extract_channel_patch(
                data_idx, sample_idx, [channel], coords, spatial_shape
            ).ravel()
            observation = observation_extractor.extract_channel_patch(
                data_idx, sample_idx, [channel], coords, spatial_shape
            ).ravel()
            pair_parts.append(
                np.column_stack((signal, observation)).astype(np.float32, copy=False)
            )
    return np.concatenate(pair_parts, axis=0)


def _yield_in_memory_batches(
    pairs: NDArray[np.float32],
    *,
    batch_size: int,
    n_epochs: int,
    lower_clip: float,
    upper_clip: float,
) -> Iterator[NDArray[np.float32]]:
    """Yield shuffled batches from an in-memory pair array.

    Parameters
    ----------
    pairs : numpy.ndarray
        Paired signal and observation intensities.
    batch_size : int
        Number of pairs per batch.
    n_epochs : int
        Number of batches to yield.
    lower_clip : float
        Lower clipping percentile.
    upper_clip : float
        Upper clipping percentile.

    Yields
    ------
    numpy.ndarray
        A training batch of intensity pairs.
    """
    signal = pairs[:, 0]
    lower_bound, upper_bound = np.percentile(signal, [lower_clip, upper_clip])
    pairs = pairs[(signal > lower_bound) & (signal < upper_bound)]
    if pairs.shape[0] == 0:
        raise ValueError(
            "No signal-observation pairs remain after percentile clipping."
        )

    np.random.shuffle(pairs)
    counter = 0
    for _ in range(n_epochs):
        if (counter + 1) * batch_size >= pairs.shape[0]:
            counter = 0
            np.random.shuffle(pairs)
        yield pairs[counter * batch_size : (counter + 1) * batch_size]
        counter += 1


def _yield_sampled_batches(
    signal_extractor: PatchExtractor[Any],
    observation_extractor: PatchExtractor[Any],
    *,
    channel: int,
    batch_size: int,
    n_epochs: int,
    lower_clip: float,
    upper_clip: float,
) -> Iterator[NDArray[np.float32]]:
    """Yield decile-sampled batches from sequential source patches.

    Parameters
    ----------
    signal_extractor : PatchExtractor
        Signal patch extractor.
    observation_extractor : PatchExtractor
        Observation patch extractor.
    channel : int
        Channel selected for training.
    batch_size : int
        Number of pairs per full batch.
    n_epochs : int
        Maximum number of batches.
    lower_clip : float
        Lower clipping percentile.
    upper_clip : float
        Upper clipping percentile.

    Yields
    ------
    numpy.ndarray
        A full or final partial training batch.
    """
    max_pixels = batch_size * n_epochs
    n_pixels = _number_of_pixels(signal_extractor.shapes)
    patch_pixels = min(batch_size * 10, math.ceil(n_pixels / n_epochs))
    samples_per_band = math.ceil(batch_size / 10)
    buffered = np.empty((0, 2), dtype=np.float32)
    yielded_pixels = 0

    for data_idx, shape in enumerate(signal_extractor.shapes):
        spatial_shape = tuple(shape[2:])
        patch_size = _large_patch_size(spatial_shape, patch_pixels)
        coord_ranges = [
            range(0, size, patch)
            for size, patch in zip(spatial_shape, patch_size, strict=True)
        ]
        for sample_idx in range(shape[0]):
            for coords in itertools.product(*coord_ranges):
                edge_patch_size = tuple(
                    min(patch, size - coord)
                    for patch, size, coord in zip(
                        patch_size, spatial_shape, coords, strict=True
                    )
                )
                signal = signal_extractor.extract_channel_patch(
                    data_idx, sample_idx, [channel], coords, edge_patch_size
                ).ravel()
                observation = observation_extractor.extract_channel_patch(
                    data_idx, sample_idx, [channel], coords, edge_patch_size
                ).ravel()
                pairs = np.column_stack((signal, observation)).astype(
                    np.float32, copy=False
                )
                sampled = _sample_intensity_deciles(
                    pairs,
                    samples_per_band=samples_per_band,
                    lower_clip=lower_clip,
                    upper_clip=upper_clip,
                )
                if sampled.shape[0] == 0:
                    continue
                buffered = np.concatenate((buffered, sampled), axis=0)

                while buffered.shape[0] >= batch_size:
                    remaining = max_pixels - yielded_pixels
                    if remaining <= 0:
                        return
                    batch_length = min(batch_size, remaining)
                    yield buffered[:batch_length]
                    buffered = buffered[batch_length:]
                    yielded_pixels += batch_length

    if buffered.shape[0] > 0 and yielded_pixels < max_pixels:
        remaining = max_pixels - yielded_pixels
        yield buffered[:remaining]


def _large_patch_size(
    spatial_shape: Sequence[int], patch_pixels: int
) -> tuple[int, ...]:
    """Choose a contiguous patch containing up to ten batches of pixels.

    Parameters
    ----------
    spatial_shape : sequence of int
        Spatial shape of an image stack.
    patch_pixels : int
        Target maximum pixels in a patch.

    Returns
    -------
    tuple of int
        Patch size for each spatial dimension.
    """
    remaining = patch_pixels
    patch_size = [1] * len(spatial_shape)
    for dim in range(len(spatial_shape) - 1, -1, -1):
        patch_size[dim] = min(spatial_shape[dim], max(1, remaining))
        remaining = math.ceil(remaining / patch_size[dim])
    return tuple(patch_size)


def _sample_intensity_deciles(
    pairs: NDArray[np.float32],
    *,
    samples_per_band: int,
    lower_clip: float,
    upper_clip: float,
) -> NDArray[np.float32]:
    """Sample up to an equal number of pairs from each signal decile.

    Parameters
    ----------
    pairs : numpy.ndarray
        Paired signal and observation intensities.
    samples_per_band : int
        Maximum pairs selected from each decile.
    lower_clip : float
        Lower clipping percentile.
    upper_clip : float
        Upper clipping percentile.

    Returns
    -------
    numpy.ndarray
        Sampled intensity pairs in random order.
    """
    signal = pairs[:, 0]
    percentile_edges = np.percentile(signal, np.linspace(lower_clip, upper_clip, 11))
    eligible = (signal > percentile_edges[0]) & (signal < percentile_edges[-1])
    band_indices = np.searchsorted(percentile_edges[1:-1], signal, side="right")

    selected: list[NDArray[np.intp]] = []
    for band in range(10):
        indices = np.flatnonzero(eligible & (band_indices == band))
        if indices.size > samples_per_band:
            indices = np.random.choice(indices, size=samples_per_band, replace=False)
        selected.append(indices)

    selected_indices = np.concatenate(selected)
    np.random.shuffle(selected_indices)
    return pairs[selected_indices]
