import copy
import tempfile
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import pytest
import tifffile

from careamics.config import Configuration
from careamics.config.algorithm import Algorithm
from careamics.config.data import Data
from careamics.config.training import LrScheduler, Optimizer, Training


@pytest.fixture
def image_size() -> Tuple[int, int]:
    return (128, 128)


@pytest.fixture
def patch_size() -> Tuple[int, int]:
    return (64, 64)


@pytest.fixture
def overlaps() -> Tuple[int, int]:
    return (32, 32)


@pytest.fixture
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
    # create dictionary
    configuration = {
        "experiment_name": "LevitatingFrog",
        "working_directory": str(tmp_path),
        "algorithm": {
            "algorithm_type": "n2v",
            "loss": "n2v",
            "model": {
                "architecture": "UNet",
                "parameters": {"depth": 2, "num_channels_init": 32},
            },
            "is_3D": False,
            "masking_strategy": {
                "strategy_type": "default",
                "parameters": {"masked_pixel_percentage": 0.2, "roi_size": 11},
            },
        },
        "training": {
            "num_epochs": 666,
            "batch_size": 42,
            "patch_size": [64, 64],
            "optimizer": {
                "name": "Adam",
            },
            "lr_scheduler": {"name": "ReduceLROnPlateau"},
            "augmentation": True,
        },
        "data": {
            "in_memory": True,
            "data_format": "tif",
            "axes": "SYX",
        },
    }

    return configuration


@pytest.fixture
def complete_config(minimum_config: dict) -> dict:
    """Create a complete configuration.

    This configuration should not be used for testing an Engine.

    Parameters
    ----------
    minimum_config : dict
        A minimum configuration.

    Returns
    -------
    dict
        A complete configuration example.
    """
    # add to configuration
    complete_config = copy.deepcopy(minimum_config)
    complete_config["algorithm"]["loss"] = "pn2v"
    complete_config["algorithm"]["noise_model"] = {
        "model_type": "hist",
        "parameters": {
            "min_value": 350,
            "max_value": 6500,
            "bins": 256,
        },
    }
    complete_config["algorithm"]["transforms"] = {
        "Flip": None,
        "ManipulateN2V": {
            "masked_pixel_percentage": 0.6,
            "roi_size": 13,
        },
    }

    complete_config["algorithm"]["model"] = {
        "architecture": "UNet",
        "parameters": {
            "depth": 8,
            "num_channels_init": 32,
        },
    }

    complete_config["training"]["optimizer"]["parameters"] = {
        "lr": 0.00999,
    }
    complete_config["training"]["lr_scheduler"]["parameters"] = {
        "patience": 22,
    }
    complete_config["training"]["use_wandb"] = True
    complete_config["training"]["num_workers"] = 6
    complete_config["training"]["amp"] = {
        "use": True,
        "init_scale": 512,
    }
    complete_config["data"]["in_memory"] = False
    complete_config["data"]["mean"] = 666.666
    complete_config["data"]["std"] = 42.420

    return complete_config


@pytest.fixture
def ordered_array() -> Callable:
    """A function that returns an array with ordered values."""

    def _ordered_array(shape: tuple, dtype=int) -> np.ndarray:
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
        return np.arange(np.prod(shape), dtype=dtype).reshape(shape)

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
    return np.arange(2048).reshape((1, 8, 16, 16))


@pytest.fixture
def temp_dir() -> Path:
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def example_data_path(
    temp_dir: Path, image_size: Tuple[int, int], patch_size: Tuple[int, int]
) -> Tuple[Path, Path]:
    test_image = np.random.rand(*image_size)

    train_path = temp_dir / "train"
    val_path = temp_dir / "val"
    test_path = temp_dir / "test"
    train_path.mkdir()
    val_path.mkdir()
    test_path.mkdir()

    tifffile.imwrite(train_path / "train_image.tif", test_image)
    tifffile.imwrite(val_path / "val_image.tif", test_image)
    tifffile.imwrite(test_path / "test_image.tif", test_image)

    return train_path, val_path, test_path


@pytest.fixture
def base_configuration(temp_dir: Path, patch_size) -> Configuration:
    configuration = Configuration(
        experiment_name="smoke_test",
        working_directory=temp_dir,
        algorithm=Algorithm(
            algorithm_type="n2v",
            loss="n2v",
            model={"architecture": "UNet"},
            is_3D="False",
            transforms={"Flip": None, "ManipulateN2V": None},
        ),
        data=Data(
            in_memory=True,
            data_format="tif",
            axes="YX",
        ),
        training=Training(
            num_epochs=1,
            patch_size=patch_size,
            batch_size=2,
            optimizer=Optimizer(name="Adam"),
            lr_scheduler=LrScheduler(name="ReduceLROnPlateau"),
            extraction_strategy="random",
            augmentation=True,
            num_workers=0,
            use_wandb=False,
        ),
    )
    return configuration


@pytest.fixture
def supervised_configuration(temp_dir: Path, patch_size) -> Configuration:
    configuration = Configuration(
        experiment_name="smoke_test",
        working_directory=temp_dir,
        algorithm=Algorithm(
            algorithm_type="n2n",
            loss="mae",
            model={"architecture": "UNet"},
            is_3D="False",
            transforms={"Flip": None},
        ),
        data=Data(
            in_memory=True,
            data_format="tif",
            axes="YX",
        ),
        training=Training(
            num_epochs=1,
            patch_size=patch_size,
            batch_size=2,
            optimizer=Optimizer(name="Adam"),
            lr_scheduler=LrScheduler(name="ReduceLROnPlateau"),
            extraction_strategy="random",
            augmentation=True,
            num_workers=0,
            use_wandb=False,
        ),
    )
    return configuration
