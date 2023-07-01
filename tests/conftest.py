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
def minimum_config(tmp_path: Path) -> dict:
    """Create a minimum configuration.

    Parameters
    ----------
    tmp_path : Path
        Temporary path for testing.

    Returns
    -------
    dict
        A minumum configuration example.
    """
    # create data in the temporary folder
    path_train = tmp_path / "training"
    create_tiff(path_train, n_files=3)

    # create dictionary
    configuration = {
        "experiment_name": "LevitatingFrog",
        "working_directory": str(tmp_path),
        "algorithm": {
            "loss": "n2v",
            "model": "UNet",
            "is_3D": False,
        },
        "training": {
            "num_epochs": 666,
            "batch_size": 42,
            "patch_size": [64, 64],
            "optimizer": {
                "name": "Adam",
            },
            "lr_scheduler": {"name": "ReduceLROnPlateau"},
            "extraction_strategy": "random",
            "augmentation": True,
        },
        "data": {
            "training_path": str(path_train),
            "data_format": "tif",
            "axes": "SYX",
        },
    }

    return configuration


@pytest.fixture
def complete_config(tmp_path: Path, minimum_config: dict) -> dict:
    """Create a complete configuration.

    Parameters
    ----------
    tmp_path : Path
        Temporary path for testing.
    minimum_config : dict
        A minimum configuration.

    Returns
    -------
    dict
        A complete configuration example.
    """

    # create model
    model = "model.pth"
    path_model = tmp_path / model
    path_model.touch()

    # create validation and prediction data
    path_validation = tmp_path / "validation"
    create_tiff(path_validation, n_files=1)

    path_test = tmp_path / "test"
    create_tiff(path_test, n_files=2)

    # add to configuration
    complete_config = minimum_config
    complete_config["trained_model"] = model

    complete_config["algorithm"]["masking_strategy"] = "default"
    complete_config["algorithm"]["masked_pixel_percentage"] = 0.2
    complete_config["algorithm"]["model_parameters"] = {
        "depth": 8,
        "num_filter_base": 128,
    }

    complete_config["training"]["optimizer"]["parameters"] = {
        "lr": 0.00999,
    }
    complete_config["training"]["lr_scheduler"]["parameters"] = {
        "patience": 22,
    }
    complete_config["training"]["num_workers"] = (6,)
    complete_config["training"]["amp"] = {
        "use": True,
        "init_scale": 1024,
    }
    complete_config["data"]["validation_path"] = str(path_validation)
    complete_config["data"]["prediction_path"] = str(path_test)

    complete_config["prediction"] = {
        "tile_shape": [64, 64],
        "overlaps": [32, 32],
    }

    return complete_config


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
