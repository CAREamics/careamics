from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from careamics import CAREamist, Configuration
from careamics.config.support import SupportedData
from careamics.model_io import export_to_bmz

# TODO add details about where each of these fixture is used (e.g. smoke test)
# TODO move each fixture to relevant conftest in subfolders


# Allows CI to run on macos-latest gh runner
@pytest.fixture(autouse=True)
def disable_mps(monkeypatch):
    """Disable MPS for all tests"""
    monkeypatch.setattr("torch._C._mps_is_available", lambda: False)


@pytest.fixture
def gaussian_likelihood_params():
    return {"predict_logvar": "pixelwise", "logvar_lowerbound": -5}


@pytest.fixture
def create_tiff(path: Path, n_files: int):
    """Create tiff files for testing."""
    if not path.exists():
        path.mkdir()

    for i in range(n_files):
        file_path = path / f"file_{i}.tif"
        file_path.touch()


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


# TODO: wrong! need to update/remove this fixture
@pytest.fixture
def minimum_algorithm_musplit() -> dict:
    """Create a minimum algorithm dictionary.

    Returns
    -------
    dict
        A minimum algorithm example.
    """
    # create dictionary
    algorithm = {
        "algorithm_type": "vae",
        "algorithm": "musplit",  # TODO temporary
        "loss": "musplit",
        "model": {
            "architecture": "LVAE",
            "z_dims": (128, 128, 128),
            "multiscale_count": 2,
            "predict_logvar": "pixelwise",
        },
        "likelihood": {
            "type": "GaussianLikelihoodConfig",
        },
    }

    return algorithm


# TODO: wrong! need to update/remove this fixture
@pytest.fixture
def minimum_algorithm_denoisplit() -> dict:
    """Create a minimum algorithm dictionary.

    Returns
    -------
    dict
        A minimum algorithm example.
    """
    # create dictionary
    algorithm = {
        "algorithm_type": "vae",
        "algorithm": "denoisplit",
        "loss": "denoisplit",
        "model": {
            "architecture": "LVAE",
            "z_dims": (128, 128, 128),
            "multiscale_count": 2,
        },
        "likelihood": {"type": "GaussianLikelihoodConfig", "color_channels": 2},
        "noise_model": "MultiChannelNMConfig",
    }

    return algorithm


@pytest.fixture
def minimum_algorithm_microsplit() -> dict:
    """Create a minimum MicroSplit algorithm dictionary.

    Returns
    -------
    dict
        A minimum MicroSplit algorithm example.
    """
    # create dictionary
    algorithm = {
        "algorithm": "microsplit",
        "loss": "microsplit",
        "model": {
            "architecture": "LVAE",
            "z_dims": (128, 128, 128),
            "multiscale_count": 2,
            "predict_logvar": "pixelwise",
        },
        "likelihood": {
            "type": "GaussianLikelihoodConfig",
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
        "lightning_trainer_config": {"max_epochs": 1},
    }

    return training


@pytest.fixture
def minimum_n2v_configuration(
    minimum_algorithm_n2v: dict, minimum_data: dict, minimum_training: dict
) -> dict:
    """Create a minimum configuration dictionary.

    Parameters
    ----------
    tmp_path : Path
        Temporary path for testing.
    minimum_algorithm : dict
        Minimum algorithm configuration.
    minimum_data_n2v : dict
        Minimum N2V data configuration.
    minimum_training : dict
        Minimum training configuration.

    Returns
    -------
    dict
        A minimum configuration example.
    """
    # update masked pixel percentage so config validation will pass
    if "n2v_config" not in minimum_algorithm_n2v:
        minimum_algorithm_n2v["n2v_config"] = {}
    mask_pixel_perc = 100 / np.prod(minimum_data["patch_size"])
    mask_pixel_perc = np.ceil(mask_pixel_perc * 10) / 10
    minimum_algorithm_n2v["n2v_config"]["masked_pixel_percentage"] = 100 / np.prod(
        minimum_data["patch_size"]
    )

    # create dictionary
    configuration = {
        "experiment_name": "LevitatingFrog",
        "algorithm_config": minimum_algorithm_n2v,
        "training_config": minimum_training,
        "data_config": minimum_data,
    }

    return configuration


@pytest.fixture
def minimum_supervised_configuration(
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
def patch_size() -> tuple[int, int]:
    return (64, 64)


@pytest.fixture
def overlaps() -> tuple[int, int]:
    return (32, 32)


@pytest.fixture
def pre_trained(tmp_path, minimum_n2v_configuration):
    """Fixture to create a pre-trained CAREamics model."""
    # training data
    train_array = np.arange(32 * 32).reshape((32, 32)).astype(np.float32)

    # create configuration
    config = Configuration(**minimum_n2v_configuration)
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
    predicted_output = careamist.predict(train_array, tta_transforms=False)
    predicted = np.concatenate(predicted_output, axis=0)

    # export to BioImage Model Zoo
    path = tmp_path / "model.zip"
    export_to_bmz(
        model=careamist.model,
        config=careamist.cfg,
        path_to_archive=path,
        model_name="TopModel",
        general_description="A model that just walked in.",
        data_description="My data.",
        authors=[{"name": "Amod", "affiliation": "El"}],
        input_array=train_array[np.newaxis, np.newaxis, ...],
        output_array=predicted,
    )
    assert path.exists()

    return path


@pytest.fixture
def create_dummy_noise_model(
    n_gaussians: int = 3,
    n_coeffs: int = 3,
) -> None:
    weights = np.random.rand(3 * n_gaussians, n_coeffs)
    nm_dict = {
        "trained_weight": weights,
        "min_signal": np.array([0]),
        "max_signal": np.array([2**16 - 1]),
        "min_sigma": 0.125,
    }
    return nm_dict


@pytest.fixture
def minimum_lvae_params():
    return {
        "input_shape": (64, 64),
        "output_channels": 2,
        "multiscale_count": 1,
        "encoder_conv_strides": [2, 2],
        "decoder_conv_strides": [2, 2],
        "z_dims": [128, 128, 128, 128],
        "encoder_n_filters": 64,
        "decoder_n_filters": 64,
        "encoder_dropout": 0.1,
        "decoder_dropout": 0.1,
        "nonlinearity": "ELU",
        "predict_logvar": "pixelwise",
        "analytical_kl": False,
    }
