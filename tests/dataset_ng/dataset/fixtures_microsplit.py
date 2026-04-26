"""Shared fixtures for MicroSplit old-vs-new equivalence tests.

This module provides:
- Deterministic synthetic 2-channel image data (N, H, W, C=2) for the old dataset
  and the same data reshaped to (S, C, Y, X) for the new pipeline.
- A factory function to create a configured `LCMultiChDloader` from synthetic data.
- Per-channel and one-mu-std statistics helpers that match the legacy computation.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

SEED = 42
N_FRAMES = 4
HEIGHT = 128
WIDTH = 128
N_CHANNELS = 2
PATCH_SIZE = 64
GRID_SIZE = 64  # grid == patch → no overlap (simplest tiling)
MULTISCALE_COUNT = 3  # multiscale_lowres_count
AXES = "SYX"  # new pipeline axes string


# ── Synthetic data helpers ─────────────────────────────────────────────────────


def make_synthetic_nhwc() -> np.ndarray:
    """Return deterministic float32 array shaped (N, H, W, C) for old dataset.

    Returns
    -------
    np.ndarray
        Synthetic data in NHWC layout.
    """
    rng = np.random.default_rng(SEED)
    # Use values well above 0 so the downsampled-ratio sanity check passes
    data = rng.uniform(10.0, 200.0, (N_FRAMES, HEIGHT, WIDTH, N_CHANNELS)).astype(
        np.float32
    )
    return data


def make_synthetic_scyx() -> np.ndarray:
    """Return deterministic float32 array shaped (S, C, Y, X) for new dataset.

    This is the same data as make_synthetic_nhwc() but transposed to the
    SC(Z)YX axis order expected by InMemoryImageStack.

    Returns
    -------
    np.ndarray
        Synthetic data in SCYX layout.
    """
    nhwc = make_synthetic_nhwc()
    # (N, H, W, C) → (S, C, Y, X)
    return nhwc.transpose(0, 3, 1, 2).copy()


# ── Legacy dataset factory ─────────────────────────────────────────────────────


class LegacyDatasetBundle(NamedTuple):
    """Holds the legacy dataset and its pre-computed statistics."""

    dataset: object  # LCMultiChDloader
    mean_dict: dict
    std_dict: dict


def make_legacy_dataset(
    multiscale_count: int = MULTISCALE_COUNT,
    enable_random_cropping: bool = True,
    uncorrelated_channels: bool = False,
    input_is_sum: bool = False,
) -> LegacyDatasetBundle:
    """Create and configure an ``LCMultiChDloader`` from synthetic data.

    Parameters
    ----------
    multiscale_count : int
        Number of LC scales (passed as ``multiscale_lowres_count``).
    enable_random_cropping : bool
        ``True`` for training-like random patches; ``False`` for deterministic.
    uncorrelated_channels : bool
        Enable uncorrelated channel mixing.
    input_is_sum : bool
        If ``True``, input is the sum of channels rather than the average.

    Returns
    -------
    LegacyDatasetBundle
        Dataset and its mean/std statistics.
    """
    from careamics.lvae_training.dataset import (
        LCMultiChDloader,
        MicroSplitDataConfig,
    )
    from careamics.lvae_training.dataset.types import (
        DataSplitType,
        DataType,
        TilingMode,
    )

    synthetic_data = make_synthetic_nhwc()

    def _load_data_fn(data_config, datapath, datasplit_type, **kwargs):
        """Return the in-memory synthetic array for legacy loader tests.

        Parameters
        ----------
        data_config : object
            Legacy data configuration (unused by this test loader).
        datapath : pathlib.Path
            Input path from the legacy loader API (unused).
        datasplit_type : object
            Requested data split value from the legacy loader API (unused).
        **kwargs : object
            Additional keyword arguments accepted for API compatibility.

        Returns
        -------
        np.ndarray
            Synthetic NHWC array used to build the legacy dataset.
        """
        return synthetic_data

    config = MicroSplitDataConfig(
        data_type=DataType.HTLIF24Data,
        axes=AXES,
        image_size=(PATCH_SIZE, PATCH_SIZE),
        grid_size=GRID_SIZE,
        num_channels=N_CHANNELS,
        batch_size=64,
        datasplit_type=DataSplitType.Train,
        multiscale_lowres_count=multiscale_count,
        tiling_mode=TilingMode.ShiftBoundary,
        enable_random_cropping=enable_random_cropping,
        uncorrelated_channels=uncorrelated_channels,
        uncorrelated_channel_probab=0.5,
        input_is_sum=input_is_sum,
        target_separate_normalization=True,
        use_one_mu_std=True,
        normalized_input=True,
        train_dataloader_params={"num_workers": 0, "shuffle": True},
        val_dataloader_params={"num_workers": 0},
    )

    dataset = LCMultiChDloader(
        data_config=config,
        datapath=Path("/synthetic"),
        load_data_fn=_load_data_fn,
    )

    mean_dict, std_dict = dataset.compute_mean_std()
    dataset.set_mean_std(mean_dict, std_dict)

    return LegacyDatasetBundle(dataset=dataset, mean_dict=mean_dict, std_dict=std_dict)


# ── Statistics extraction helpers ─────────────────────────────────────────────


def compute_legacy_stats(
    data: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Compute legacy-style mean/std dictionaries from (N,H,W,C) data.

    Replicates the ``MultiChDloader.compute_mean_std`` logic with
    ``target_separate_normalization=True``, ``use_one_mu_std=True``,
    ``input_is_sum=False`` (canonical path).

    Parameters
    ----------
    data : np.ndarray
        Array of shape (N, H, W, C) as used by the legacy dataset.

    Returns
    -------
    tuple[dict[str, np.ndarray], dict[str, np.ndarray]]
        ``(mean_dict, std_dict)`` in the same structure as
        ``compute_mean_std``.
    """
    # One-mu-std input stats: single mean/std across the entire array, repeated C times
    input_mean = np.mean(data).reshape(1, 1, 1, 1)
    input_std = np.std(data).reshape(1, 1, 1, 1)
    input_mean = np.repeat(input_mean, N_CHANNELS, axis=1)
    input_std = np.repeat(input_std, N_CHANNELS, axis=1)

    # Per-channel target stats
    target_means = []
    target_stds = []
    for ch in range(data.shape[-1]):
        target_means.append(data[..., ch].mean())
        target_stds.append(data[..., ch].std())
    target_mean = np.array(target_means)[None, :, None, None]  # (1, C, 1, 1)
    target_std = np.array(target_stds)[None, :, None, None]

    return (
        {"input": input_mean, "target": target_mean},
        {"input": input_std, "target": target_std},
    )
