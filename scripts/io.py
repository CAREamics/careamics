"""IO helpers for MicroSplit inference scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray


def list_files(
    datadir: str | Path,
    split: Literal["train", "val", "test"],
    subset: Literal["inputs", "targets"],
) -> list[Path]:
    """Return sorted file paths under `<datadir>/<subset>/<split>/`."""
    files_dir = Path(datadir) / subset / split
    if not files_dir.is_dir():
        raise FileNotFoundError(files_dir)
    return sorted(p for p in files_dir.iterdir() if p.is_file())


def npz_key(source: str, data_idx: int) -> str:
    """Pick an NPZ archive key for a stitched prediction.

    For file-backed inputs the key is `Path(source).stem` so each prediction can
    be matched to its source by filename. For in-memory array inputs (where
    `InMemoryImageStack` reports `source="array"`) the key falls back to a
    zero-padded `data_idx` so multiple in-memory inputs don't collide.

    Parameters
    ----------
    source : str
        The `ImageRegionData.source` value carried by the tile.
    data_idx : int
        Index of the input image in the prediction dataset.

    Returns
    -------
    str
        Key suitable for `np.savez_compressed(..., **{key: array})`.
    """
    if source == "array":
        return f"pred_{data_idx:04d}"
    return Path(source).stem


def save_predictions_npz(
    results: dict[str, NDArray],
    save_dir: str | Path,
    filename: str = "predictions.npz",
) -> Path:
    """Write `results` as a single compressed NPZ.

    Parameters
    ----------
    results : dict of {str: NDArray}
        Per-image predictions keyed by archive key (see :func:`npz_key`).
    save_dir : str or Path
        Output directory. Created if missing.
    filename : str, default="predictions.npz"
        NPZ filename inside `save_dir`.

    Returns
    -------
    Path
        Absolute path to the written file.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / filename
    np.savez_compressed(out_path, **results)
    return out_path


def save_inference_params(
    params: dict[str, Any],
    save_dir: str | Path,
    filename: str = "inference_params.json",
) -> Path:
    """Write the main inference parameters as a JSON sidecar.

    Parameters
    ----------
    params : dict of {str: Any}
        Inference parameters (e.g. tile size, overlap, mmse_count, checkpoint
        and data paths). `Path` values are serialized as strings.
    save_dir : str or Path
        Output directory. Created if missing.
    filename : str, default="inference_params.json"
        JSON filename inside `save_dir`.

    Returns
    -------
    Path
        Absolute path to the written file.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / filename
    with out_path.open("w") as f:
        json.dump(params, f, indent=2, default=str)
    return out_path
