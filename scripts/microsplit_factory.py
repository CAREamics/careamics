"""MicroSplit Lightning-module factory for inference scripts."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from careamics.dev.sliding_window_tiled_pred import load_microsplit_from_checkpoint

from .config_factory import create_algorithm_config, pkl_load

if TYPE_CHECKING:
    import torch

    from careamics.lightning.modules.microsplit_module import MicroSplitModule


def build_microsplit_module(
    ckpt_path: str | Path,
    pkl_path: str | Path,
    *,
    mmse_count: int = 1,
    device: "torch.device | str | None" = None,
) -> "MicroSplitModule":
    """Instantiate a `MicroSplitModule` and load weights from a checkpoint.

    Reads the architecture + loss + likelihood configuration from `pkl_path`
    (legacy `config.pkl`) via :func:`scripts.config_factory.create_algorithm_config`
    and loads weights from `ckpt_path`.

    Note: target-channel denormalization stats are *not* set here — call
    `module.set_target_stats(...)` after building the prediction dataset.

    Parameters
    ----------
    ckpt_path : str or Path
        Path to the Lightning checkpoint file
        (e.g. `<ckpt_dir>/BaselineVAECL_best.ckpt`).
    pkl_path : str or Path
        Path to the legacy training-config dump (e.g. `<ckpt_dir>/config.pkl`).
    mmse_count : int, default=1
        Number of stochastic forward passes per tile at predict time. `1` is
        canonical for `SlidingWindowTiledPatching`; for classical inner tiling
        raise to ~50.
    device : torch.device or str or None, default=None
        If provided, the module is moved to this device.

    Returns
    -------
    MicroSplitModule
        Module in eval mode with weights loaded.
    """
    pkl_data = pkl_load(pkl_path)
    algorithm_config = create_algorithm_config(pkl_data)
    algorithm_config.mmse_count = mmse_count

    module = load_microsplit_from_checkpoint(
        ckpt_path=Path(ckpt_path),
        algorithm_config=algorithm_config,
    )
    module.eval()
    if device is not None:
        module = module.to(device)
    return module
