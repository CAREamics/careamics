import tempfile
from pathlib import Path
from typing import Callable, Generator, Tuple

import numpy as np
import pytest
import tifffile

from careamics.config import Configuration
from careamics.config.algorithm_model import (
    AlgorithmModel,
    LrSchedulerModel,
    OptimizerModel,
)
from careamics.config.data_model import DataModel
from careamics.config.support import SupportedData
from careamics.config.training_model import TrainingModel


# TODO add details about where each of these fixture is used (e.g. smoke test)
@pytest.fixture
def create_tiff(path: Path, n_files: int):
    """Create tiff files for testing."""
    if not path.exists():
        path.mkdir()

    for i in range(n_files):
        file_path = path / f"file_{i}.tif"
        file_path.touch()


@pytest.fixture
def minimum_algorithm() -> dict:
    """Create a minimum algorithm dictionary.

    Returns
    -------
    dict
        A minimum algorithm example.
    """
    # create dictionary
    algorithm = {
        "algorithm": "custom",
        "loss": "n2v",
        "model": {
            "architecture": "UNet",
        },
    }

    return algorithm


@pytest.fixture
def minimum_data() -> dict:
    """Create a minimum data dictionary.

    Returns
    -------
    dict
        A minimum data example.
    """
    # create dictionary
    data = {
        "data_type": SupportedData.TIFF.value,
        "patch_size": [64, 64],
        "axes": "SYX",
    }

    return data


@pytest.fixture
def minimum_prediction() -> dict:
    """Create a minimum prediction dictionary.

    Returns
    -------
    dict
        A minimum data example.
    """
    # create dictionary
    predic = {
        "data_type": SupportedData.TIFF.value,
        "tile_size": [64, 64],
        "axes": "SYX",
    }

    return predic


@pytest.fixture
def minimum_training() -> dict:
    """Create a minimum training dictionary.

    Returns
    -------
    dict
        A minimum training example.
    """
    # create dictionary
    training = {
        "num_epochs": 666,
    }

    return training


@pytest.fixture
def minimum_configuration(
    minimum_algorithm: dict, minimum_data: dict, minimum_training: dict
) -> dict:
    """Create a minimum configuration dictionary.

    Parameters
    ----------
    tmp_path : Path
        Temporary path for testing.
    minimum_algorithm : dict
        Minimum algorithm configuration.
    minimum_data : dict
        Minimum data configuration.
    minimum_training : dict
        Minimum training configuration.

    Returns
    -------
    dict
        A minumum configuration example.
    """
    # create dictionary
    configuration = {
        "experiment_name": "LevitatingFrog",
        "algorithm": minimum_algorithm,
        "training": minimum_training,
        "data": minimum_data,
    }

    return configuration


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
    """A 2D array with shape (1, 3, 10, 9).

    Returns
    -------
    np.ndarray
        2D array with shape (1, 3, 10, 9).
    """
    return np.arange(90 * 3).reshape((1, 3, 10, 9))


@pytest.fixture
def array_3D() -> np.ndarray:
    """A 3D array with shape (1, 3, 5, 10, 9).

    Returns
    -------
    np.ndarray
        3D array with shape (1, 3, 5, 10, 9).
    """
    return np.arange(2048 * 3).reshape((1, 3, 8, 16, 16))


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


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
        algorithm=AlgorithmModel(
            algorithm="n2v",
            loss="n2v",
            model={"architecture": "UNet"},
            is_3D="False",
            transforms={"Flip": None, "ManipulateN2V": None},
        ),
        data=DataModel(
            in_memory=True,
            extension="tif",
            axes="YX",
        ),
        training=TrainingModel(
            num_epochs=1,
            patch_size=patch_size,
            batch_size=2,
            optimizer=OptimizerModel(name="Adam"),
            lr_scheduler=LrSchedulerModel(name="ReduceLROnPlateau"),
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
        algorithm=AlgorithmModel(
            algorithm="n2n",
            loss="mae",
            model={"architecture": "UNet"},
            is_3D="False",
            transforms={"Flip": None},
        ),
        data=DataModel(
            in_memory=True,
            extension="tif",
            axes="YX",
        ),
        training=TrainingModel(
            num_epochs=1,
            patch_size=patch_size,
            batch_size=2,
            optimizer=OptimizerModel(name="Adam"),
            lr_scheduler=LrSchedulerModel(name="ReduceLROnPlateau"),
            extraction_strategy="random",
            augmentation=True,
            num_workers=0,
            use_wandb=False,
        ),
    )
    return configuration
