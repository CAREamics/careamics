from pathlib import Path
from typing import Callable

import numpy as np
import pytest

# TODO Make two fixtures, one full configuration and one minimal


def create_tiff(path: Path, n_files: int):
    """Create tiff files for testing."""
    if not path.exists():
        path.mkdir()

    for i in range(n_files):
        file_path = path / f"file_{i}.tif"
        file_path.touch()


@pytest.fixture
def test_config(tmp_path) -> dict:
    # create data
    ext = "tif"

    path_train = tmp_path / "train"
    create_tiff(path_train, n_files=3)

    path_validation = tmp_path / "validation"
    create_tiff(path_validation, n_files=1)

    path_test = tmp_path / "test"
    create_tiff(path_test, n_files=2)

    # create dictionary
    test_configuration = {
        "run_params": {
            "experiment_name": "testing",
            "workdir": str(tmp_path),
        },
        "algorithm": {
            "loss": ["n2v"],
            "model": "UNet",
            "num_masked_pixels": 0.2,
            "pixel_manipulation": "n2v",
        },
        "training": {
            "num_epochs": 100,
            "learning_rate": 0.0001,
            "optimizer": {
                "name": "Adam",
                "parameters": {
                    "lr": 0.001,
                    "betas": [0.9, 0.999],
                    "eps": 1e-08,
                    "weight_decay": 0.0005,
                    "amsgrad": True,
                },
            },
            "lr_scheduler": {
                "name": "ReduceLROnPlateau",
                "parameters": {"factor": 0.5, "patience": 5, "mode": "min"},
            },
            "amp": {
                "toggle": False,
                "init_scale": 1024,
            },
            "data": {
                "path": str(path_train),
                "ext": ext,
                "axes": "YX",
                "extraction_strategy": "sequential",
                "patch_size": [128, 128],
                "batch_size": 1,
            },
        },
        "evaluation": {
            "data": {
                "path": str(path_validation),
                "ext": ext,
                "axes": "YX",
                "extraction_strategy": "sequential",
                "patch_size": [128, 128],
                "batch_size": 1,
            },
            "metric": "psnr",
        },
        "prediction": {
            "data": {
                "path": str(path_test),
                "ext": ext,
                "axes": "YX",
                "extraction_strategy": "sequential",
                "patch_size": [128, 128],
                "batch_size": 1,
            },
            "overlap": [25, 25],
        },
    }

    return test_configuration


@pytest.fixture
def ordered_array() -> Callable:
    """A function that returns an array with ordered values."""

    def _ordered_array(shape: tuple) -> np.ndarray:
        """An array with ordered values.

        Parameters
        ----------
        shape : tuple
            Shape of the array.

        Returns
        -------
        np.ndarray
            Array with ordered values.
        """
        return np.arange(np.prod(shape)).reshape(shape)

    return _ordered_array


@pytest.fixture
def array_2D() -> np.ndarray:
    """A 2D array with shape (1, 10, 9).

    Returns
    -------
    np.ndarray
        2D array with shape (1, 10, 9).
    """
    return np.arange(90).reshape((1, 10, 9))


@pytest.fixture
def array_3D() -> np.ndarray:
    """A 3D array with shape (1, 5, 10, 9).

    Returns
    -------
    np.ndarray
        3D array with shape (1, 5, 10, 9).
    """
    return np.arange(450).reshape((1, 5, 10, 9))
