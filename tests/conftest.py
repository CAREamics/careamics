from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import pytest

from careamics import CAREamist, Configuration
from careamics.config.support import SupportedData
from careamics.model_io import export_to_bmz


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
def minimum_algorithm_custom() -> dict:
    """Create a minimum algorithm dictionary.

    Returns
    -------
    dict
        A minimum algorithm example.
    """
    # create dictionary
    algorithm = {
        "algorithm": "custom",
        "loss": "mae",
        "model": {
            "architecture": "UNet",
        },
    }

    return algorithm


@pytest.fixture
def minimum_algorithm_n2v() -> dict:
    """Create a minimum algorithm dictionary.

    Returns
    -------
    dict
        A minimum algorithm example.
    """
    # create dictionary
    algorithm = {
        "algorithm": "n2v",
        "loss": "n2v",
        "model": {
            "architecture": "UNet",
        },
    }

    return algorithm


@pytest.fixture
def minimum_algorithm_supervised() -> dict:
    """Create a minimum algorithm dictionary.

    Returns
    -------
    dict
        A minimum algorithm example.
    """
    # create dictionary
    algorithm = {
        "algorithm": "n2n",
        "loss": "mae",
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
        "data_type": SupportedData.ARRAY.value,
        "patch_size": [8, 8],
        "axes": "YX",
    }

    return data


@pytest.fixture
def minimum_inference() -> dict:
    """Create a minimum inference dictionary.

    Returns
    -------
    dict
        A minimum data example.
    """
    # create dictionary
    predic = {
        "data_type": SupportedData.ARRAY.value,
        "axes": "YX",
        "image_means": [2.0],
        "image_stds": [1.0],
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
        "num_epochs": 1,
    }

    return training


@pytest.fixture
def minimum_configuration(
    minimum_algorithm_n2v: dict, minimum_data: dict, minimum_training: dict
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
        "algorithm_config": minimum_algorithm_n2v,
        "training_config": minimum_training,
        "data_config": minimum_data,
    }

    return configuration


@pytest.fixture
def supervised_configuration(
    minimum_algorithm_supervised: dict, minimum_data: dict, minimum_training: dict
) -> dict:
    configuration = {
        "experiment_name": "LevitatingFrog",
        "algorithm_config": minimum_algorithm_supervised,
        "training_config": minimum_training,
        "data_config": minimum_data,
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
        return np.arange(np.prod(shape), dtype=dtype).reshape(shape).astype(np.float32)

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
def patch_size() -> Tuple[int, int]:
    return (64, 64)


@pytest.fixture
def overlaps() -> Tuple[int, int]:
    return (32, 32)


@pytest.fixture
def pre_trained(tmp_path, minimum_configuration):
    """Fixture to create a pre-trained CAREamics model."""
    # training data
    train_array = np.arange(32 * 32).reshape((32, 32)).astype(np.float32)

    # create configuration
    config = Configuration(**minimum_configuration)
    config.training_config.num_epochs = 1
    config.data_config.axes = "YX"
    config.data_config.batch_size = 2
    config.data_config.data_type = SupportedData.ARRAY.value
    config.data_config.patch_size = (8, 8)

    # instantiate CAREamist
    careamist = CAREamist(source=config, work_dir=tmp_path)

    # train CAREamist
    careamist.train(train_source=train_array)

    # check that it trained
    pre_trained_path: Path = tmp_path / "checkpoints" / "last.ckpt"
    assert pre_trained_path.exists()

    return pre_trained_path


@pytest.fixture
def pre_trained_bmz(tmp_path, pre_trained) -> Path:
    """Fixture to create a BMZ model."""
    # training data
    train_array = np.ones((32, 32), dtype=np.float32)

    # instantiate CAREamist
    careamist = CAREamist(source=pre_trained, work_dir=tmp_path)

    # predict (no tiling and no tta)
    predicted = careamist.predict(train_array, tta_transforms=False)

    # export to BioImage Model Zoo
    path = tmp_path / "model.zip"
    export_to_bmz(
        model=careamist.model,
        config=careamist.cfg,
        path=path,
        name="TopModel",
        general_description="A model that just walked in.",
        authors=[{"name": "Amod", "affiliation": "El"}],
        input_array=train_array[np.newaxis, np.newaxis, ...],
        output_array=predicted,
    )
    assert path.exists()

    return path
