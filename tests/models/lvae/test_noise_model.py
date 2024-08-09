from pathlib import Path

import pytest
import numpy as np
import torch

def create_dummy_noise_model(
    tmp_path: Path,
    n_gaussians: int,
    n_coeffs: int,
) -> None:
    weights = np.random.rand(3*n_gaussians, n_coeffs)
    nm_dict = {
        "trained_weight": weights,
        "min_signal": np.array([0]),
        "max_signal": np.array([2**16 - 1]),
        "min_sigma": 0.125,
    }
    np.savez(tmp_path / "dummy_noise_model.npz", **nm_dict)
    

