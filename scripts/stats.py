"""Normalization-stat computation for MicroSplit experiments.

Checkpoints in `<CKPT_ROOT>/<exp>/` carry no normalization stats, so we recompute
them from the training data — once per experiment — and cache the result in a
JSON sidecar next to the data. The sidecar is invalidated automatically when the
training files change (detected via a cheap name+size+mtime hash).

At inference time there is no on-the-fly input synthesis — every experiment has
real input files on disk under `inputs/<split>/*.tif` and real target files
under `targets/<split>/*.tif`. We therefore compute stats independently from
each directory using the same per-channel pipeline, regardless of how the
inputs got there at train time. (Note: for legacy multiplexed experiments where
the trainer synthesised inputs on the fly from targets, this means the input
*scale* used here may differ from what the model saw during training. The model
is robust to scale shifts via BatchNorm in both encoder and decoder, but if you
see degraded predictions on multiplexed datasets this is a likely first thing
to investigate.)

The returned dict shape matches `MeanStdConfig`:

    {"input_means": [...], "input_stds": [...],
     "target_means": [...], "target_stds": [...]}

so it can be splat directly into :func:`scripts.config_factory.get_predict_config`.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import tifffile

from .io import list_files

logger = logging.getLogger(__name__)

_STATS_SCHEMA_VERSION = 1 # bump to invalidate all caches when the shape or meaning of the payload changes
_STATS_FILENAME = "stats.json"


def load_or_compute_stats(
    name: str,
    data_dir: Path,
    *,
    is_3d: bool = False,
    force: bool = False,
) -> dict[str, list[float]]:
    """Return cached normalization stats, recomputing if missing or stale.

    Parameters
    ----------
    name : str
        Human-readable identifier for the experiment / dataset, written to the
        sidecar for traceability (does not affect the cache key).
    data_dir : Path
        Experiment data directory containing `inputs/train/*.tif` and
        `targets/train/*.tif`.
    is_3d : bool, default=False
        Whether files have a Z axis. Used to canonicalize array shapes.
    force : bool, default=False
        Bypass the cache and recompute from scratch.

    Returns
    -------
    dict[str, list[float]]
        Keys: `input_means`, `input_stds`, `target_means`, `target_stds`.
        Suitable to splat into :func:`scripts.config_factory.get_predict_config`.
    """
    sidecar = Path(data_dir) / _STATS_FILENAME
    files = _files_driving_computation(data_dir)
    current_hash = _hash_files(files)

    if not force and sidecar.is_file():
        try:
            cached = json.loads(sidecar.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Could not read cached stats at %s (%s) — recomputing.",
                sidecar,
                exc
            )
        else:
            if (
                cached.get("version") == _STATS_SCHEMA_VERSION
                and cached.get("train_files_hash") == current_hash
            ):
                return _strip_metadata(cached)
            logger.warning(
                "Cached stats at %s are stale (train files changed or schema "
                "bumped) — recomputing.",
                sidecar,
            )

    stats = _compute_stats(data_dir, is_3d=is_3d)

    payload = {
        "version": _STATS_SCHEMA_VERSION,
        "name": name,
        "train_files_hash": current_hash,
        **stats,
    }
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote normalization stats to %s", sidecar)
    return stats


def _compute_stats(data_dir: Path, *, is_3d: bool) -> dict[str, list[float]]:
    """Per-channel stats from `inputs/train/*` and `targets/train/*`."""
    input_files = list_files(data_dir, "train", "inputs")
    target_files = list_files(data_dir, "train", "targets")

    input_acc = _PerChannelWelford()
    for f in input_files:
        arr = _load_canonical(f, is_3d=is_3d)
        input_acc.update(arr)

    target_acc = _PerChannelWelford()
    for f in target_files:
        arr = _load_canonical(f, is_3d=is_3d)
        target_acc.update(arr)

    input_means, input_stds = input_acc.finalize()
    target_means, target_stds = target_acc.finalize()
    return {
        "input_means": input_means,
        "input_stds": input_stds,
        "target_means": target_means,
        "target_stds": target_stds,
    }


class _ScalarWelford:
    """Streaming scalar mean/var (Welford's algorithm, numerically stable)."""

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0  # sum of squared deviations from the running mean

    def update(self, values: np.ndarray) -> None:
        chunk = np.asarray(values, dtype=np.float64).reshape(-1)
        n_b = chunk.size
        if n_b == 0:
            return
        mean_b = float(chunk.mean())
        m2_b = float(((chunk - mean_b) ** 2).sum())

        n_a = self.count
        n = n_a + n_b
        delta = mean_b - self.mean
        self.mean += delta * (n_b / n)
        self.m2 += m2_b + delta * delta * (n_a * n_b / n)
        self.count = n

    def finalize(self) -> tuple[float, float]:
        if self.count == 0:
            raise ValueError("No data accumulated.")
        var = self.m2 / self.count
        return self.mean, float(np.sqrt(max(var, 0.0)))


class _PerChannelWelford:
    """Streaming per-channel mean/var with the channel axis at index 1.

    Inputs are expected to have axes `(S, C, [Z], Y, X)`. The first call fixes C;
    subsequent calls must match.
    """

    def __init__(self) -> None:
        self._scalars: list[_ScalarWelford] | None = None

    def update(self, arr: np.ndarray) -> None:
        # arr has shape (S, C, ...). Per-channel update treats every non-C axis
        # as a sample for that channel.
        n_channels = arr.shape[1]
        if self._scalars is None:
            self._scalars = [_ScalarWelford() for _ in range(n_channels)]
        elif len(self._scalars) != n_channels:
            raise ValueError(
                f"Inconsistent channel count: previously saw "
                f"{len(self._scalars)}, now {n_channels}."
            )
        # iterate channels; reshape per-channel slice to 1D and update
        moved = np.moveaxis(arr, 1, 0)  # (C, S, ..., Y, X)
        flat = moved.reshape(n_channels, -1)
        for c in range(n_channels):
            self._scalars[c].update(flat[c])

    def finalize(self) -> tuple[list[float], list[float]]:
        if self._scalars is None:
            raise ValueError("No data accumulated.")
        means, stds = zip(*(s.finalize() for s in self._scalars), strict=True)
        return list(means), list(stds)
    

def _load_canonical(path: Path, *, is_3d: bool) -> np.ndarray:
    """Load a TIFF and reshape to `(S, C, [Z], Y, X)`.

    Resolves layout heuristically: any missing leading axes (S, C) are prepended
    as size-1 axes. The function accepts files with ndim in
    `{spatial, spatial+1, spatial+2}` where `spatial` is 2 (2D) or 3 (3D).

    Files with non-matching ndim raise `ValueError`.
    """
    arr = np.asarray(tifffile.imread(path))
    spatial = 3 if is_3d else 2
    target_ndim = spatial + 2  # S, C, plus spatial
    extra = target_ndim - arr.ndim
    if extra < 0 or extra > 2:
        raise ValueError(
            f"Unexpected ndim {arr.ndim} for {'3D' if is_3d else '2D'} TIFF "
            f"{path}: expected one of {{spatial, spatial+1, spatial+2}} "
            f"= {{{spatial}, {spatial+1}, {spatial+2}}}."
        )
    for _ in range(extra):
        arr = arr[np.newaxis]
    return arr


def _files_driving_computation(data_dir: Path) -> list[Path]:
    """Files whose contents determine the stats (for cache invalidation)."""
    return list_files(data_dir, "train", "inputs") + list_files(
        data_dir, "train", "targets"
    )


def _hash_files(files: list[Path]) -> str:
    """sha1 of sorted `"<name>:<size>:<mtime_ns>"` for cache invalidation.

    Hashes file metadata rather than contents — cheap and sufficient to detect
    train-set changes (renames, replacements, additions).
    """
    parts = sorted(
        f"{p.name}:{p.stat().st_size}:{p.stat().st_mtime_ns}" for p in files
    )
    h = hashlib.sha1()
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"\n")
    return "sha1:" + h.hexdigest()


def _strip_metadata(cached: dict) -> dict[str, list[float]]:
    """Keep only the four MeanStdConfig keys from a cached payload."""
    return {
        k: cached[k]
        for k in ("input_means", "input_stds", "target_means", "target_stds")
    }
