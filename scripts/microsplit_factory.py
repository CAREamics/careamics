"""MicroSplit Lightning-module factory for inference scripts."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from careamics.lightning.modules.microsplit_module import MicroSplitModule

from .config_factory import create_algorithm_config, pkl_load

if TYPE_CHECKING:
    pass


# Top-level keys that don't map onto the NG MicroSplitModule and are dropped
# before loading. `noiseModel.*` and `likelihood_NM.*` carry the noise-model +
# NM-likelihood weights that v1 doesn't load (gaussian likelihood only — see
# `scripts.config_factory.get_likelihood_config`).
_DROP_KEY_PREFIXES: tuple[str, ...] = ("noiseModel.", "likelihood_NM.")

# Suffixes dropped from each key. `num_batches_tracked` is a BatchNorm
# bookkeeping buffer that PyTorch saves by default but the NG LVAE's BN layers
# don't register, so it shows up as "unexpected" at load time.
_DROP_KEY_SUFFIXES: tuple[str, ...] = (".num_batches_tracked",)


def convert_legacy_state_dict(state_dict: dict) -> dict:
    """Rewrite a legacy MicroSplit checkpoint state-dict into NG layout.

    Two transforms:
    1. Prepend the ``model.`` prefix to every retained key. Legacy checkpoints
       store the raw ``LVAE`` ``nn.Module``'s keys at the root; the NG
       :class:`MicroSplitModule` wraps the LVAE as ``self.model``, so every
       architecture key needs the prefix.
    2. Drop keys that don't belong to the NG model: noise-model + NM-likelihood
       weights (we use Gaussian likelihood only in v1), and `num_batches_tracked`
       BN buffers (not registered by the NG BN layers).

    Parameters
    ----------
    state_dict : dict
        Raw state-dict pulled out of `ckpt["state_dict"]`.

    Returns
    -------
    dict
        State-dict with NG-compatible keys; safe to pass to
        :meth:`MicroSplitModule.load_state_dict` with ``strict=True``.
    """
    converted: dict = {}
    for key, value in state_dict.items():
        if any(key.startswith(p) for p in _DROP_KEY_PREFIXES):
            continue
        if any(key.endswith(s) for s in _DROP_KEY_SUFFIXES):
            continue
        converted[f"model.{key}"] = value
    return converted


def build_microsplit_module(
    ckpt_path: str | Path,
    pkl_path: str | Path,
    *,
    mmse_count: int = 1,
    device: "torch.device | str | None" = None,
) -> MicroSplitModule:
    """Instantiate a `MicroSplitModule` and load weights from a checkpoint.

    Reads the architecture + loss + likelihood configuration from `pkl_path`
    (legacy `config.pkl`) via
    :func:`scripts.config_factory.create_algorithm_config`, and loads weights
    from `ckpt_path` after running the state-dict through
    :func:`convert_legacy_state_dict` to repair the legacy → NG key layout.

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

    module = MicroSplitModule(algorithm_config=algorithm_config)

    ckpt = torch.load(
        Path(ckpt_path), map_location="cpu", weights_only=False
    )
    state_dict = convert_legacy_state_dict(ckpt["state_dict"])
    module.load_state_dict(state_dict, strict=True)

    module.eval()
    if device is not None:
        module = module.to(device)
    return module
