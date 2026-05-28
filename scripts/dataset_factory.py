"""MicroSplit prediction-dataset factory.

Single entry point that glues together:
- normalization-stat computation (with caching) from :mod:`scripts.stats`
- legacy training-config loading via :func:`scripts.config_factory.pkl_load`
- prediction-config building via :func:`scripts.config_factory.get_predict_config`
- the NG MicroSplit prediction dataset factory

into a single call:

    >>> from pathlib import Path
    >>> from scripts.dataset_factory import build_pred_dataset
    >>> dataset = build_pred_dataset(
    ...     data_dir=Path("/.../switi/data/HT_LIF24_5ms"),
    ...     pkl_path=Path("/.../switi/ckpts/HT_LIF24_5ms/config.pkl"),
    ...     split="test",
    ...     stride=[4, 4],
    ... )

The caller wraps the returned dataset in a `DataLoader` (so it can tune
`num_workers` / `pin_memory` / etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from careamics.dataset.dataset import CareamicsDataset
from careamics.dataset.factory.microsplit_factory import create_microsplit_pred_dataset

from .config_factory import get_predict_config, pkl_load
from .io import list_files
from .stats import load_or_compute_stats


def build_pred_dataset(
    data_dir: str | Path,
    pkl_path: str | Path,
    *,
    name: str | None = None,
    split: Literal["train", "val", "test"] = "test",
    overlap: list[int],
    stride: list[int] | None = None,
    batch_size: int = 1,
    force_recompute_stats: bool = False,
) -> CareamicsDataset:
    """Build a MicroSplit prediction dataset for a single experiment.

    Parameters
    ----------
    data_dir : str or Path
        Experiment data directory laid out as
        `<data_dir>/{inputs,targets}/{train,val,test}/*.tif`.
    pkl_path : str or Path
        Path to the legacy MicroSplit training config dump for this experiment
        (typically `<ckpt_dir>/config.pkl`). Used to recover patch size,
        multiscale count, padding mode, and 2D-vs-3D mode.
    name : str or None, default=None
        Human-readable identifier written to the stats sidecar for provenance.
        If `None`, falls back to `Path(data_dir).name`.
    split : {"train", "val", "test"}, default="test"
        Which on-disk split to predict on.
    overlap : list of int
        Tile overlap per spatial dimension (length 2 for 2D, length 3 for 3D).
    stride : list of int or None, default=None
        If `None`, classical inner-tiling (`TiledPatchingConfig`) is used.
        Otherwise sliding-window inner-tiling
        (`SlidingWindowTiledPatchingConfig`) with the given stride.
    batch_size : int, default=1
        Prediction batch size, baked into the returned dataset's config.
    force_recompute_stats : bool, default=False
        If `True`, bypass the `<data_dir>/stats.json` cache.

    Returns
    -------
    CareamicsDataset
        Ready-to-iterate prediction dataset. Wrap in a `DataLoader` to use.
    """
    data_dir = Path(data_dir)
    pkl_path = Path(pkl_path)
    name = name if name is not None else data_dir.name

    pkl_root = pkl_load(pkl_path)
    # Legacy configs may nest the data section under "data" or be flat.
    pkl_data = pkl_root.get("data", pkl_root)
    is_3d = bool(pkl_data.get("mode_3D", False))

    stats = load_or_compute_stats(
        name,
        data_dir,
        is_3d=is_3d,
        force=force_recompute_stats,
    )

    config = get_predict_config(
        pkl_data,
        overlap=overlap,
        stride=stride,
        batch_size=batch_size,
        **stats,
    )

    input_files = list_files(data_dir, split, "inputs")
    return create_microsplit_pred_dataset(config, input_data=input_files)
