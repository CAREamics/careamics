from pathlib import Path
from threading import Thread

import numpy as np
import pytest
import tifffile
from numpy.typing import NDArray
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint

from careamics import CAREamist
from careamics.config import Configuration, save_configuration
from careamics.config.support import SupportedAlgorithm, SupportedData
from careamics.dataset.dataset_utils import reshape_array
from careamics.lightning.callbacks import HyperParametersCallback, ProgressBarCallback
from careamics.lightning.predict_data_module import create_predict_datamodule


def random_array(shape: tuple[int, ...], seed: int = 42):
    """Return a random array with values between 0 and 255."""
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 255, shape)).astype(np.float32)


def test_no_parameters():
    """Test that CAREamics cannot be instantiated without parameters."""
    with pytest.raises(TypeError):
        CAREamist()


def test_minimum_configuration_via_object(
    tmp_path: Path, minimum_n2v_configuration: dict
):
    """Test that CAREamics can be instantiated with a minimum configuration object."""
    # create configuration
    config = Configuration(**minimum_n2v_configuration)

    # instantiate CAREamist
    CAREamist(source=config, work_dir=tmp_path)


def test_minimum_configuration_via_path(
    tmp_path: Path, minimum_n2v_configuration: dict
):
    """Test that CAREamics can be instantiated with a path to a minimum
    configuration.
    """
    # create configuration
    config = Configuration(**minimum_n2v_configuration)
    path_to_config = save_configuration(config, tmp_path)

    # instantiate CAREamist
    CAREamist(source=path_to_config)


def test_train_error_target_unsupervised_algorithm(
    tmp_path: Path, minimum_n2v_configuration: dict
):
    """Test that an error is raised when a target is provided for N2V."""
    # create configuration
    config = Configuration(**minimum_n2v_configuration)
    config.algorithm_config.algorithm = SupportedAlgorithm.N2V.value

    # train error with Paths
    config.data_config.data_type = SupportedData.TIFF.value
    careamics = CAREamist(source=config, work_dir=tmp_path)
    with pytest.raises(ValueError):
        careamics.train(
            train_source=tmp_path,
            train_target=tmp_path,
        )

    # train error with strings
    with pytest.raises(ValueError):
        careamics.train(
            train_source=str(tmp_path),
            train_target=str(tmp_path),
        )

    # train error with arrays
    config.data_config.data_type = SupportedData.ARRAY.value
    careamics = CAREamist(source=config, work_dir=tmp_path)
    with pytest.raises(ValueError):
        careamics.train(
            train_source=np.ones((32, 32)),
            train_target=np.ones((32, 32)),
        )


@pytest.mark.mps_gh_fail
def test_train_single_array_no_val(tmp_path: Path, minimum_n2v_configuration: dict):
    """Test that CAREamics can be trained with arrays."""
    # training data
    train_array = random_array((32, 32))

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
    assert Path(tmp_path / "checkpoints" / "last.ckpt").exists()

    # export to BMZ
    careamist.export_to_bmz(
        path_to_archive=tmp_path / "model.zip",
        friendly_model_name="TopModel",
        input_array=train_array,
        authors=[{"name": "Amod", "affiliation": "El"}],
        general_description="A model that just walked in.",
        data_description="A random array.",
    )
    assert (tmp_path / "model.zip").exists()


@pytest.mark.mps_gh_fail
def test_train_array(tmp_path: Path, minimum_n2v_configuration: dict):
    """Test that CAREamics can be trained on arrays."""
    # training data
    train_array = random_array((32, 32))
    val_array = random_array((32, 32))

    # create configuration
    config = Configuration(**minimum_n2v_configuration)
    config.data_config.axes = "YX"
    config.data_config.batch_size = 2
    config.data_config.data_type = SupportedData.ARRAY.value
    config.data_config.patch_size = (8, 8)

    # Test save_top_k=2 functionality
    config.training_config.lightning_trainer_config = {"max_epochs": 6}
    config.training_config.checkpoint_callback = {
        "save_top_k": 2,
        "monitor": "val_loss",
        "mode": "min",
        "save_last": True,
        "every_n_epochs": 1,
    }

    # instantiate CAREamist
    careamist = CAREamist(source=config, work_dir=tmp_path)

    # train CAREamist
    careamist.train(train_source=train_array, val_source=val_array)

    # check that it trained
    assert Path(tmp_path / "checkpoints" / "last.ckpt").exists()

    # check save_top_k=2 functionality
    # Should have: 2 best checkpoints + 1 last.ckpt = 3 total files
    checkpoint_files = list((tmp_path / "checkpoints").glob("*.ckpt"))
    assert (
        len(checkpoint_files) == 3
    ), f"Expected 3 checkpoint files (2 best + last), found {len(checkpoint_files)}"

    # Verify last.ckpt exists
    last_ckpt_exists = any(f.name == "last.ckpt" for f in checkpoint_files)
    assert last_ckpt_exists, "last.ckpt should exist when save_last=True"

    # Verify we have exactly 2 non-last checkpoint files
    non_last_checkpoints = [f for f in checkpoint_files if f.name != "last.ckpt"]
    assert (
        len(non_last_checkpoints) == 2
    ), f"Expected exactly 2 best checkpoints, found {len(non_last_checkpoints)}"

    # export to BMZ
    careamist.export_to_bmz(
        path_to_archive=tmp_path / "model.zip",
        friendly_model_name="TopModel",
        input_array=train_array,
        authors=[{"name": "Amod", "affiliation": "El"}],
        general_description="A model that just walked in.",
        data_description="A random array.",
    )
    assert (tmp_path / "model.zip").exists()


@pytest.mark.mps_gh_fail
@pytest.mark.parametrize("independent_channels", [False, True])
def test_train_array_channel(
    tmp_path: Path, minimum_n2v_configuration: dict, independent_channels: bool
):
    """Test that CAREamics can be trained on arrays with channels."""
    # training data
    train_array = random_array((32, 32, 3))
    val_array = random_array((32, 32, 3))

    # create configuration
    config = Configuration(**minimum_n2v_configuration)
    config.data_config.axes = "YXC"
    config.algorithm_config.model.in_channels = 3
    config.algorithm_config.model.num_classes = 3
    config.algorithm_config.model.independent_channels = independent_channels
    config.data_config.batch_size = 2
    config.data_config.data_type = SupportedData.ARRAY.value
    config.data_config.patch_size = (8, 8)

    # instantiate CAREamist
    careamist = CAREamist(source=config, work_dir=tmp_path)

    # train CAREamist
    careamist.train(train_source=train_array, val_source=val_array)

    # check that it trained
    assert Path(tmp_path / "checkpoints" / "last.ckpt").exists()

    # export to BMZ
    careamist.export_to_bmz(
        path_to_archive=tmp_path / "model.zip",
        friendly_model_name="TopModel",
        input_array=train_array,
        authors=[{"name": "Amod", "affiliation": "El"}],
        general_description="A model that just walked in.",
        data_description="A random array.",
        channel_names=["red", "green", "blue"],
    )
    assert (tmp_path / "model.zip").exists()


@pytest.mark.mps_gh_fail
def test_train_array_3d(tmp_path: Path, minimum_n2v_configuration: dict):
    """Test that CAREamics can be trained on 3D arrays."""
    # training data
    train_array = random_array((8, 32, 32))
    val_array = random_array((8, 32, 32))

    # create configuration
    minimum_n2v_configuration["data_config"]["axes"] = "ZYX"
    minimum_n2v_configuration["data_config"]["patch_size"] = (8, 16, 16)
    config = Configuration(**minimum_n2v_configuration)
    config.data_config.batch_size = 2
    config.data_config.data_type = SupportedData.ARRAY.value

    # instantiate CAREamist
    careamist = CAREamist(source=config, work_dir=tmp_path)

    # train CAREamist
    careamist.train(train_source=train_array, val_source=val_array)

    # check that it trained
    assert Path(tmp_path / "checkpoints" / "last.ckpt").exists()

    # export to BMZ
    careamist.export_to_bmz(
        path_to_archive=tmp_path / "model.zip",
        friendly_model_name="TopModel",
        input_array=train_array,
        authors=[{"name": "Amod", "affiliation": "El"}],
        general_description="A model that just walked in.",
        data_description="A random array.",
    )
    assert (tmp_path / "model.zip").exists()


@pytest.mark.mps_gh_fail
def test_train_tiff_files_in_memory_no_val(
    tmp_path: Path, minimum_n2v_configuration: dict
):
    """Test that CAREamics can be trained with tiff files in memory."""
    # training data
    train_array = random_array((32, 32))

    # save files
    train_file = tmp_path / "train.tiff"
    tifffile.imwrite(train_file, train_array)

    # create configuration
    config = Configuration(**minimum_n2v_configuration)
    config.data_config.axes = "YX"
    config.data_config.batch_size = 2
    config.data_config.data_type = SupportedData.TIFF.value
    config.data_config.patch_size = (8, 8)

    # instantiate CAREamist
    careamist = CAREamist(source=config, work_dir=tmp_path)

    # train CAREamist
    careamist.train(train_source=train_file)

    # check that it trained
    assert Path(tmp_path / "checkpoints" / "last.ckpt").exists()

    # export to BMZ
    careamist.export_to_bmz(
        path_to_archive=tmp_path / "model.zip",
        friendly_model_name="TopModel",
        input_array=train_array,
        authors=[{"name": "Amod", "affiliation": "El"}],
        general_description="A model that just walked in.",
        data_description="A random array.",
    )
    assert (tmp_path / "model.zip").exists()


@pytest.mark.mps_gh_fail
def test_train_tiff_files_in_memory(tmp_path: Path, minimum_n2v_configuration: dict):
    """Test that CAREamics can be trained with tiff files in memory."""
    # training data
    train_array = random_array((32, 32))
    val_array = random_array((32, 32))

    # save files
    train_file = tmp_path / "train.tiff"
    tifffile.imwrite(train_file, train_array)

    val_file = tmp_path / "val.tiff"
    tifffile.imwrite(val_file, val_array)

    # create configuration
    config = Configuration(**minimum_n2v_configuration)
    config.data_config.axes = "YX"
    config.data_config.batch_size = 2
    config.data_config.data_type = SupportedData.TIFF.value
    config.data_config.patch_size = (8, 8)

    # instantiate CAREamist
    careamist = CAREamist(source=config, work_dir=tmp_path)

    # train CAREamist
    careamist.train(train_source=train_file, val_source=val_file)

    # check that it trained
    assert Path(tmp_path / "checkpoints" / "last.ckpt").exists()

    # export to BMZ
    careamist.export_to_bmz(
        path_to_archive=tmp_path / "model.zip",
        friendly_model_name="TopModel",
        input_array=train_array,
        authors=[{"name": "Amod", "affiliation": "El"}],
        general_description="A model that just walked in.",
        data_description="A random array.",
    )
    assert (tmp_path / "model.zip").exists()


@pytest.mark.mps_gh_fail
def test_train_tiff_files(tmp_path: Path, minimum_n2v_configuration: dict):
    """Test that CAREamics can be trained with tiff files by deactivating
    the in memory dataset.
    """
    # training data
    train_array = random_array((32, 32))
    val_array = random_array((32, 32))

    # save files
    train_file = tmp_path / "train.tiff"
    tifffile.imwrite(train_file, train_array)

    val_file = tmp_path / "val.tiff"
    tifffile.imwrite(val_file, val_array)

    # create configuration
    config = Configuration(**minimum_n2v_configuration)
    config.data_config.axes = "YX"
    config.data_config.batch_size = 2
    config.data_config.data_type = SupportedData.TIFF.value
    config.data_config.patch_size = (8, 8)

    # instantiate CAREamist
    careamist = CAREamist(source=config, work_dir=tmp_path)

    # train CAREamist
    careamist.train(train_source=train_file, val_source=val_file, use_in_memory=False)

    # check that it trained
    assert Path(tmp_path / "checkpoints" / "last.ckpt").exists()

    # export to BMZ
    careamist.export_to_bmz(
        path_to_archive=tmp_path / "model.zip",
        friendly_model_name="TopModel",
        input_array=train_array,
        authors=[{"name": "Amod", "affiliation": "El"}],
        general_description="A model that just walked in.",
        data_description="A random array.",
    )
    assert (tmp_path / "model.zip").exists()


@pytest.mark.mps_gh_fail
def test_train_array_supervised(tmp_path: Path, minimum_supervised_configuration: dict):
    """Test that CAREamics can be trained with arrays."""
    # training data
    train_array = random_array((32, 32))
    val_array = random_array((32, 32))
    train_target = random_array((32, 32))
    val_target = random_array((32, 32))

    # create configuration
    config = Configuration(**minimum_supervised_configuration)
    config.data_config.axes = "YX"
    config.data_config.batch_size = 2
    config.data_config.data_type = SupportedData.ARRAY.value
    config.data_config.patch_size = (8, 8)

    # instantiate CAREamist
    careamist = CAREamist(source=config, work_dir=tmp_path)

    # train CAREamist
    careamist.train(
        train_source=train_array,
        val_source=val_array,
        train_target=train_target,
        val_target=val_target,
    )

    # check that it trained
    assert Path(tmp_path / "checkpoints" / "last.ckpt").exists()

    # export to BMZ
    careamist.export_to_bmz(
        path_to_archive=tmp_path / "model.zip",
        friendly_model_name="TopModel",
        input_array=train_array,
        authors=[{"name": "Amod", "affiliation": "El"}],
        general_description="A model that just walked in.",
        data_description="A random array.",
    )
    assert (tmp_path / "model.zip").exists()


@pytest.mark.mps_gh_fail
def test_train_tiff_files_in_memory_supervised(
    tmp_path: Path, minimum_supervised_configuration: dict
):
    """Test that CAREamics can be trained with tiff files in memory."""
    # training data
    train_array = random_array((32, 32))
    val_array = random_array((32, 32))
    train_target = random_array((32, 32))
    val_target = random_array((32, 32))

    # save files
    images = tmp_path / "images"
    images.mkdir()
    train_file = images / "train.tiff"
    tifffile.imwrite(train_file, train_array)

    val_file = tmp_path / "images" / "val.tiff"
    tifffile.imwrite(val_file, val_array)

    targets = tmp_path / "targets"
    targets.mkdir()
    train_target_file = targets / "train.tiff"
    tifffile.imwrite(train_target_file, train_target)

    val_target_file = targets / "val.tiff"
    tifffile.imwrite(val_target_file, val_target)

    # create configuration
    config = Configuration(**minimum_supervised_configuration)
    config.data_config.axes = "YX"
    config.data_config.batch_size = 2
    config.data_config.data_type = SupportedData.TIFF.value
    config.data_config.patch_size = (8, 8)

    # instantiate CAREamist
    careamist = CAREamist(source=config, work_dir=tmp_path)

    # train CAREamist
    careamist.train(
        train_source=train_file,
        val_source=val_file,
        train_target=train_target_file,
        val_target=val_target_file,
    )

    # check that it trained
    assert Path(tmp_path / "checkpoints" / "last.ckpt").exists()

    # export to BMZ
    careamist.export_to_bmz(
        path_to_archive=tmp_path / "model.zip",
        friendly_model_name="TopModel",
        input_array=train_array,
        authors=[{"name": "Amod", "affiliation": "El"}],
        general_description="A model that just walked in.",
        data_description="A random array.",
    )
    assert (tmp_path / "model.zip").exists()


@pytest.mark.mps_gh_fail
def test_train_tiff_files_supervised(
    tmp_path: Path, minimum_supervised_configuration: dict
):
    """Test that CAREamics can be trained with tiff files by deactivating
    the in memory dataset.
    """
    # training data
    train_array = random_array((32, 32))
    val_array = random_array((32, 32))
    train_target = random_array((32, 32))
    val_target = random_array((32, 32))

    # save files
    images = tmp_path / "images"
    images.mkdir()
    train_file = images / "train.tiff"
    tifffile.imwrite(train_file, train_array)

    val_file = tmp_path / "images" / "val.tiff"
    tifffile.imwrite(val_file, val_array)

    targets = tmp_path / "targets"
    targets.mkdir()
    train_target_file = targets / "train.tiff"
    tifffile.imwrite(train_target_file, train_target)

    val_target_file = targets / "val.tiff"
    tifffile.imwrite(val_target_file, val_target)

    # create configuration
    config = Configuration(**minimum_supervised_configuration)
    config.data_config.axes = "YX"
    config.data_config.batch_size = 2
    config.data_config.data_type = SupportedData.TIFF.value
    config.data_config.patch_size = (8, 8)

    # instantiate CAREamist
    careamist = CAREamist(source=config, work_dir=tmp_path)

    # train CAREamist
    careamist.train(
        train_source=train_file,
        val_source=val_file,
        train_target=train_target_file,
        val_target=val_target_file,
        use_in_memory=False,
    )

    # check that it trained
    assert Path(tmp_path / "checkpoints" / "last.ckpt").exists()

    # export to BMZ
    careamist.export_to_bmz(
        path_to_archive=tmp_path / "model.zip",
        friendly_model_name="TopModel",
        input_array=train_array,
        authors=[{"name": "Amod", "affiliation": "El"}],
        general_description="A model that just walked in.",
        data_description="A random array.",
    )
    assert (tmp_path / "model.zip").exists()


@pytest.mark.mps_gh_fail
@pytest.mark.parametrize("samples", [1, 2, 4])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_predict_on_array_tiled(
    tmp_path: Path, minimum_n2v_configuration: dict, batch_size, samples
):
    """Test that CAREamics can predict on arrays."""
    # training data
    train_array = random_array((samples, 32, 32))

    # create configuration
    config = Configuration(**minimum_n2v_configuration)
    config.data_config.axes = "SYX"
    config.data_config.batch_size = 2
    config.data_config.data_type = SupportedData.ARRAY.value
    config.data_config.patch_size = (8, 8)

    # instantiate CAREamist
    careamist = CAREamist(source=config, work_dir=tmp_path)

    # train CAREamist
    careamist.train(train_source=train_array)

    # predict CAREamist
    predicted = careamist.predict(
        train_array, batch_size=batch_size, tile_size=(16, 16), tile_overlap=(4, 4)
    )

    assert (
        np.concatenate(predicted).shape
        == reshape_array(train_array, config.data_config.axes).shape
    )

    # export to BMZ
    careamist.export_to_bmz(
        path_to_archive=tmp_path / "model.zip",
        friendly_model_name="TopModel",
        input_array=train_array,
        authors=[{"name": "Amod", "affiliation": "El"}],
        general_description="A model that just walked in.",
        data_description="A random array.",
    )
    assert (tmp_path / "model.zip").exists()


@pytest.mark.mps_gh_fail
@pytest.mark.parametrize("samples", [1, 2, 4])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_predict_arrays_no_tiling(
    tmp_path: Path, minimum_n2v_configuration: dict, batch_size, samples
):
    """Test that CAREamics can predict on arrays without tiling."""
    # training data
    train_array = random_array((samples, 32, 32))

    # create configuration
    config = Configuration(**minimum_n2v_configuration)
    config.data_config.axes = "SYX"
    config.data_config.batch_size = 2
    config.data_config.data_type = SupportedData.ARRAY.value
    config.data_config.patch_size = (8, 8)

    # instantiate CAREamist
    careamist = CAREamist(source=config, work_dir=tmp_path)

    # train CAREamist
    careamist.train(train_source=train_array)

    # predict CAREamist
    predicted = careamist.predict(train_array, batch_size=batch_size)

    assert (
        np.concatenate(predicted).shape
        == reshape_array(train_array, config.data_config.axes).shape
    )

    # export to BMZ
    careamist.export_to_bmz(
        path_to_archive=tmp_path / "model.zip",
        friendly_model_name="TopModel",
        input_array=train_array,
        authors=[{"name": "Amod", "affiliation": "El"}],
        general_description="A model that just walked in.",
        data_description="A random array.",
    )
    assert (tmp_path / "model.zip").exists()


@pytest.mark.skip(
    reason=(
        "This might be a problem at the PyTorch level during `forward`. Values up to "
        "0.001 different."
    )
)
def test_batched_prediction(tmp_path: Path, minimum_n2v_configuration: dict):
    "Compare outputs when a batch size of 1 or 2 is used"

    tile_size = (16, 16)
    tile_overlap = (4, 4)
    shape = (32, 32)

    train_array = random_array(shape)
    # create configuration
    config = Configuration(**minimum_n2v_configuration)
    config.data_config.axes = "YX"
    config.data_config.batch_size = 2
    config.data_config.data_type = SupportedData.ARRAY.value

    # instantiate CAREamist
    careamist = CAREamist(source=config, work_dir=tmp_path)

    # train CAREamist
    careamist.train(train_source=train_array)

    # predict with batch size 1 and batch size 2
    pred_bs_1 = careamist.predict(
        train_array, batch_size=1, tile_size=tile_size, tile_overlap=tile_overlap
    )
    pred_bs_2 = careamist.predict(
        train_array, batch_size=2, tile_size=tile_size, tile_overlap=tile_overlap
    )

    assert np.array_equal(pred_bs_1, pred_bs_2)


@pytest.mark.mps_gh_fail
@pytest.mark.parametrize("independent_channels", [False, True])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_predict_tiled_channel(
    tmp_path: Path,
    minimum_n2v_configuration: dict,
    independent_channels: bool,
    batch_size: int,
):
    """Test that CAREamics can be trained on arrays with channels."""
    # training data
    train_array = random_array((3, 32, 32))
    val_array = random_array((3, 32, 32))

    # create configuration
    config = Configuration(**minimum_n2v_configuration)
    config.data_config.axes = "CYX"
    config.algorithm_config.model.in_channels = 3
    config.algorithm_config.model.num_classes = 3
    config.algorithm_config.model.independent_channels = independent_channels
    config.data_config.batch_size = batch_size
    config.data_config.data_type = SupportedData.ARRAY.value
    config.data_config.patch_size = (8, 8)

    # instantiate CAREamist
    careamist = CAREamist(source=config, work_dir=tmp_path)

    # train CAREamist
    careamist.train(train_source=train_array, val_source=val_array)

    # predict CAREamist
    predicted = careamist.predict(
        train_array, batch_size=batch_size, tile_size=(16, 16), tile_overlap=(4, 4)
    )

    assert (
        np.concatenate(predicted).shape
        == reshape_array(train_array, config.data_config.axes).shape
    )


@pytest.mark.mps_gh_fail
@pytest.mark.parametrize("tiled", [True, False])
@pytest.mark.parametrize("n_samples", [1, 2])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_predict_path(
    tmp_path: Path, minimum_n2v_configuration: dict, batch_size, n_samples, tiled
):
    """Test that CAREamics can predict with tiff files."""
    # training data
    train_array = random_array((32, 32))

    # save files
    for i in range(n_samples):
        train_file = tmp_path / f"train_{i}.tiff"
        tifffile.imwrite(train_file, train_array)

    # create configuration
    config = Configuration(**minimum_n2v_configuration)
    config.data_config.axes = "YX"
    config.data_config.batch_size = 2
    config.data_config.data_type = SupportedData.TIFF.value
    config.data_config.patch_size = (8, 8)

    # instantiate CAREamist
    careamist = CAREamist(source=config, work_dir=tmp_path)

    # train CAREamist
    careamist.train(train_source=train_file)

    if tiled:
        tile_size = (16, 16)
        tile_overlap = (4, 4)
    else:
        tile_size = None
        tile_overlap = None

    # predict CAREamist
    predicted = careamist.predict(
        train_file,
        batch_size=batch_size,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
    )

    # check that it predicted
    if isinstance(predicted, list):
        for p in predicted:
            assert p.squeeze().shape == train_array.shape
    else:
        assert predicted.squeeze().shape == train_array.shape

    # export to BMZ
    careamist.export_to_bmz(
        path_to_archive=tmp_path / "model.zip",
        friendly_model_name="TopModel",
        input_array=train_array,
        authors=[{"name": "Amod", "affiliation": "El"}],
        general_description="A model that just walked in.",
        data_description="A random array.",
    )
    assert (tmp_path / "model.zip").exists()


@pytest.mark.mps_gh_fail
def test_predict_pretrained_checkpoint(tmp_path: Path, pre_trained: Path):
    """Test that CAREamics can be instantiated with a pre-trained network and predict
    on an array."""
    # prediction data
    source_array = random_array((32, 32))

    # instantiate CAREamist
    careamist = CAREamist(source=pre_trained, work_dir=tmp_path)
    assert careamist.cfg.data_config.image_means is not None
    assert careamist.cfg.data_config.image_stds is not None

    # predict
    predicted = careamist.predict(source_array)

    # check that it predicted
    assert (
        np.concatenate(predicted).shape
        == reshape_array(source_array, careamist.cfg.data_config.axes).shape
    )


@pytest.mark.mps_gh_fail
def test_predict_pretrained_bmz(tmp_path: Path, pre_trained_bmz: Path):
    """Test that CAREamics can be instantiated with a BMZ archive and predict."""
    # prediction data
    source_array = random_array((32, 32))

    # instantiate CAREamist
    careamist = CAREamist(source=pre_trained_bmz, work_dir=tmp_path)

    # predict
    predicted = careamist.predict(source_array)

    # check that it predicted
    assert (
        np.concatenate(predicted).shape
        == reshape_array(source_array, careamist.cfg.data_config.axes).shape
    )


@pytest.mark.mps_gh_fail
def test_export_bmz_pretrained_prediction(tmp_path: Path, pre_trained: Path):
    """Test that CAREamics can be instantiated with a pre-trained network and exported
    to BMZ after prediction.

    In this case, the careamist extracts the BMZ test data from the prediction
    datamodule.
    """
    # instantiate CAREamist
    careamist = CAREamist(source=pre_trained, work_dir=tmp_path)

    # prediction data
    source_array = random_array((32, 32))
    _ = careamist.predict(source_array)
    assert len(careamist.pred_datamodule.predict_dataloader()) > 0

    # export to BMZ (random array created)
    careamist.export_to_bmz(
        path_to_archive=tmp_path / "model.zip",
        friendly_model_name="TopModel",
        input_array=source_array,
        authors=[{"name": "Amod", "affiliation": "El"}],
        general_description="A model that just walked in.",
        data_description="A random array.",
    )
    assert (tmp_path / "model.zip").exists()


@pytest.mark.mps_gh_fail
def test_export_bmz_pretrained_with_array(tmp_path: Path, pre_trained: Path):
    """Test that CAREamics can be instantiated with a pre-trained network and exported
    to BMZ.

    In this case, we provide an array to the BMZ archive test.
    """
    # instantiate CAREamist
    careamist = CAREamist(source=pre_trained, work_dir=tmp_path)

    # alternatively we can pass an array
    array = random_array((32, 32))
    careamist.export_to_bmz(
        path_to_archive=tmp_path / "model2.zip",
        friendly_model_name="TopModel",
        input_array=array,
        authors=[{"name": "Amod", "affiliation": "El"}],
        general_description="A model that just walked in.",
        data_description="A random array.",
    )
    assert (tmp_path / "model2.zip").exists()


@pytest.mark.mps_gh_fail
def test_predict_to_disk_path_tiff(tmp_path, minimum_n2v_configuration):
    """Test predict_to_disk function with path source and tiff write type."""

    # prepare dummy data
    train_array = random_array((32, 32))

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    n_samples = 2
    # save files
    for i in range(n_samples):
        train_file = image_dir / f"image_{i}.tiff"
        tifffile.imwrite(train_file, train_array)

    # create configuration
    config = Configuration(**minimum_n2v_configuration)
    config.data_config.axes = "YX"
    config.data_config.batch_size = 2
    config.data_config.data_type = SupportedData.TIFF.value
    config.data_config.patch_size = (8, 8)

    # train
    careamist = CAREamist(source=config, work_dir=tmp_path)
    careamist.train(train_source=image_dir)

    # predict to disk
    careamist.predict_to_disk(source=image_dir)

    for i in range(n_samples):
        assert (tmp_path / "predictions" / f"image_{i}.tiff").is_file()


@pytest.mark.mps_gh_fail
def test_predict_to_disk_datamodule_tiff(tmp_path, minimum_n2v_configuration):
    """Test predict_to_disk function with datamodule source and tiff write type."""

    # prepare dummy data
    train_array = random_array((32, 32))

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    n_samples = 2
    # save files
    for i in range(n_samples):
        train_file = image_dir / f"image_{i}.tiff"
        tifffile.imwrite(train_file, train_array)

    # create configuration
    config = Configuration(**minimum_n2v_configuration)
    config.data_config.axes = "YX"
    config.data_config.batch_size = 2
    config.data_config.data_type = SupportedData.TIFF.value
    config.data_config.patch_size = (8, 8)

    # train
    careamist = CAREamist(source=config, work_dir=tmp_path)
    careamist.train(train_source=image_dir)

    datamodule = create_predict_datamodule(
        pred_data=image_dir,
        data_type=config.data_config.data_type,
        axes=config.data_config.axes,
        image_means=careamist.cfg.data_config.image_means,
        image_stds=careamist.cfg.data_config.image_stds,
    )

    # predict to disk
    careamist.predict_to_disk(source=datamodule)

    for i in range(n_samples):
        assert (tmp_path / "predictions" / f"image_{i}.tiff").is_file()


@pytest.mark.mps_gh_fail
def test_predict_to_disk_custom(tmp_path, minimum_n2v_configuration):
    """Test predict_to_disk function with custom write type."""

    def write_numpy(file_path: Path, img: NDArray, *args, **kwargs) -> None:
        np.save(file=file_path, arr=img)

    # prepare dummy data
    train_array = random_array((32, 32))

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    n_samples = 2
    # save files
    for i in range(n_samples):
        train_file = image_dir / f"image_{i}.tiff"
        tifffile.imwrite(train_file, train_array)

    # create configuration
    config = Configuration(**minimum_n2v_configuration)
    config.data_config.axes = "YX"
    config.data_config.batch_size = 2
    config.data_config.data_type = SupportedData.TIFF.value
    config.data_config.patch_size = (8, 8)

    # train
    careamist = CAREamist(source=config, work_dir=tmp_path)
    careamist.train(train_source=image_dir)

    # predict to disk
    careamist.predict_to_disk(
        source=image_dir,
        write_type=SupportedData.CUSTOM,
        write_extension=".npy",
        write_func=write_numpy,
    )

    for i in range(n_samples):
        assert (tmp_path / "predictions" / f"image_{i}.npy").is_file()


@pytest.mark.mps_gh_fail
def test_predict_to_disk_custom_raises(tmp_path, minimum_n2v_configuration):
    """
    Test predict_to_disk custom write type raises ValueError.

    ValueError should be raised if no write_extension or no write_func is provided.
    """

    def write_numpy(file_path: Path, img: NDArray, *args, **kwargs) -> None:
        np.save(file=file_path, arr=img)

    # prepare dummy data
    train_array = random_array((32, 32))

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    n_samples = 2
    # save files
    for i in range(n_samples):
        train_file = image_dir / f"image_{i}.tiff"
        tifffile.imwrite(train_file, train_array)

    # create configuration
    config = Configuration(**minimum_n2v_configuration)
    config.data_config.axes = "YX"
    config.data_config.batch_size = 2
    config.data_config.data_type = SupportedData.TIFF.value
    config.data_config.patch_size = (8, 8)

    # train
    careamist = CAREamist(source=config, work_dir=tmp_path)
    careamist.train(train_source=image_dir)

    with pytest.raises(ValueError):
        # no write extension provided
        careamist.predict_to_disk(
            source=image_dir,
            write_type=SupportedData.CUSTOM,
            write_extension=None,
            write_func=write_numpy,
        )
    with pytest.raises(ValueError):
        # no write func provided.
        careamist.predict_to_disk(
            source=image_dir,
            write_type=SupportedData.CUSTOM,
            write_extension=".npy",
            write_func=None,
        )


@pytest.mark.mps_gh_fail
def test_add_custom_callback(tmp_path, minimum_n2v_configuration):
    """Test that custom callback can be added to the CAREamist."""

    # define a custom callback
    class MyCallback(Callback):
        def __init__(self):
            super().__init__()

            self.has_started = False
            self.has_ended = False

        def on_train_start(self, trainer, pl_module):
            self.has_started = True

        def on_train_end(self, trainer, pl_module):
            self.has_ended = True

    my_callback = MyCallback()
    assert not my_callback.has_started
    assert not my_callback.has_ended

    # training data
    train_array = random_array((32, 32))

    # create configuration
    config = Configuration(**minimum_n2v_configuration)
    config.data_config.axes = "YX"
    config.data_config.batch_size = 2
    config.data_config.data_type = SupportedData.ARRAY.value
    config.data_config.patch_size = (8, 8)

    # instantiate CAREamist
    careamist = CAREamist(source=config, work_dir=tmp_path, callbacks=[my_callback])
    assert not my_callback.has_started
    assert not my_callback.has_ended

    # train CAREamist
    careamist.train(train_source=train_array)

    # check the state of the callback
    assert my_callback.has_started
    assert my_callback.has_ended


@pytest.mark.mps_gh_fail
@pytest.mark.parametrize("enable_progress_bar", [True, False])
def test_enable_progress_bar(
    tmp_path: Path, minimum_n2v_configuration: dict, enable_progress_bar: bool
):
    """Test no error occurs when enable_progress_bar is set to True or False."""
    # training data
    train_array = random_array((32, 32))

    # create configuration
    config = Configuration(**minimum_n2v_configuration)
    config.data_config.axes = "YX"
    config.data_config.batch_size = 2
    config.data_config.data_type = SupportedData.ARRAY.value
    config.data_config.patch_size = (8, 8)

    # instantiate CAREamist
    careamist = CAREamist(
        source=config, work_dir=tmp_path, enable_progress_bar=enable_progress_bar
    )
    is_progress_bar_callback = [
        isinstance(callback, ProgressBarCallback) for callback in careamist.callbacks
    ]
    # test progress bar has been created or not according to enable_progress_bar flag
    if enable_progress_bar:
        assert any(is_progress_bar_callback)
    else:
        assert not any(is_progress_bar_callback)

    # make sure training happens without error
    careamist.train(train_source=train_array)

    # check that it trained
    assert Path(tmp_path / "checkpoints" / "last.ckpt").exists()


def test_error_passing_careamics_callback(tmp_path, minimum_n2v_configuration):
    """Test that an error is thrown if we pass known callbacks to CAREamist."""
    # create configuration
    config = Configuration(**minimum_n2v_configuration)
    config.data_config.axes = "YX"
    config.data_config.batch_size = 2
    config.data_config.data_type = SupportedData.ARRAY.value
    config.data_config.patch_size = (8, 8)

    # Lightning callbacks
    model_ckp = ModelCheckpoint()

    with pytest.raises(ValueError):
        CAREamist(source=config, work_dir=tmp_path, callbacks=[model_ckp])

    early_stp = EarlyStopping(
        Trainer(
            max_epochs=1,
            default_root_dir=tmp_path,
        )
    )

    with pytest.raises(ValueError):
        CAREamist(source=config, work_dir=tmp_path, callbacks=[early_stp])

    # CAREamics callbacks
    progress_bar = ProgressBarCallback()

    with pytest.raises(ValueError):
        CAREamist(source=config, work_dir=tmp_path, callbacks=[progress_bar])

    hyper_params = HyperParametersCallback(config=config)

    with pytest.raises(ValueError):
        CAREamist(source=config, work_dir=tmp_path, callbacks=[hyper_params])


def test_stop_training(tmp_path: Path, minimum_n2v_configuration: dict):
    """Test that CAREamics can stop the training"""
    # training data
    train_array = random_array((32, 32))

    # create configuration
    config = Configuration(**minimum_n2v_configuration)
    config.training_config.lightning_trainer_config = {"max_epochs": 100}
    config.data_config.axes = "YX"
    config.data_config.batch_size = 2
    config.data_config.data_type = SupportedData.ARRAY.value
    config.data_config.patch_size = (8, 8)

    # instantiate CAREamist
    careamist = CAREamist(source=config, work_dir=tmp_path)

    def _train():
        careamist.train(train_source=train_array)

    # create thread
    thread = Thread(target=_train)
    thread.start()

    # stop training
    careamist.stop_training()
    thread.join()

    assert careamist.trainer.should_stop


@pytest.mark.mps_gh_fail
def test_read_logger(tmp_path, minimum_n2v_configuration):

    config = Configuration(**minimum_n2v_configuration)
    config.training_config.lightning_trainer_config = {"max_epochs": 10}

    array = np.arange(32 * 32).reshape((32, 32))

    careamist = CAREamist(config, work_dir=tmp_path)
    careamist.train(train_source=array)
    losses = careamist.get_losses()

    assert len(losses) == 4
    for key in losses:
        assert (
            len(losses[key])
            == config.training_config.lightning_trainer_config["max_epochs"]
        )


def test_all_parameters_used(tmp_path, minimum_n2v_configuration):
    config = Configuration(**minimum_n2v_configuration)
    config.training_config.lightning_trainer_config = {"max_epochs": 2}
    config.data_config.batch_size = 2
    config.data_config.patch_size = (16, 16)
    config.data_config.val_dataloader_params = {"num_workers": 2}

    # TODO add other params ?
    careamist = CAREamist(config, work_dir=tmp_path)

    assert careamist.cfg.training_config.lightning_trainer_config["max_epochs"] == 2
    assert careamist.cfg.data_config.batch_size == 2
    assert all(i == 16 for i in careamist.cfg.data_config.patch_size)
    assert careamist.cfg.data_config.val_dataloader_params["num_workers"] == 2


@pytest.mark.mps_gh_fail
def test_trainer_parameters_passed_correctly(
    tmp_path: Path, minimum_n2v_configuration: dict
):
    """Limit_train_batches and num_epochs are passed to Lightning trainer."""
    # training data
    train_array = random_array((32, 32))

    # Test limit_train_batches parameter
    limit_train_batches = 5

    # create configuration with limit_train_batches
    config = Configuration(**minimum_n2v_configuration)
    config.data_config.axes = "YX"
    config.data_config.batch_size = 2
    config.data_config.data_type = SupportedData.ARRAY.value
    config.data_config.patch_size = (8, 8)
    config.training_config.lightning_trainer_config = {
        "max_epochs": 3,
        "limit_train_batches": limit_train_batches,
    }

    # instantiate CAREamist
    careamist = CAREamist(source=config, work_dir=tmp_path)

    # Verify parameters are in the configuration
    assert (
        careamist.cfg.training_config.lightning_trainer_config["limit_train_batches"]
        == limit_train_batches
    )
    assert careamist.cfg.training_config.lightning_trainer_config["max_epochs"] == 3

    # train CAREamist (this will create the actual trainer)
    careamist.train(train_source=train_array)

    # Verify parameters were passed to the actual Lightning trainer
    assert careamist.trainer.limit_train_batches == limit_train_batches
    assert careamist.trainer.max_epochs == 3

    # check that it trained
    assert Path(tmp_path / "checkpoints" / "last.ckpt").exists()
