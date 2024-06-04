from pathlib import Path

import numpy as np
import pytest
import tifffile

from careamics import CAREamist, Configuration, save_configuration
from careamics.config.support import SupportedAlgorithm, SupportedData

# TODO test 3D and channels


def test_no_parameters():
    """Test that CAREamics cannot be instantiated without parameters."""
    with pytest.raises(TypeError):
        CAREamist()


def test_minimum_configuration_via_object(tmp_path: Path, minimum_configuration: dict):
    """Test that CAREamics can be instantiated with a minimum configuration object."""
    # create configuration
    config = Configuration(**minimum_configuration)

    # instantiate CAREamist
    CAREamist(source=config, work_dir=tmp_path)


def test_minimum_configuration_via_path(tmp_path: Path, minimum_configuration: dict):
    """Test that CAREamics can be instantiated with a path to a minimum
    configuration.
    """
    # create configuration
    config = Configuration(**minimum_configuration)
    path_to_config = save_configuration(config, tmp_path)

    # instantiate CAREamist
    CAREamist(source=path_to_config)


def test_train_error_target_unsupervised_algorithm(
    tmp_path: Path, minimum_configuration: dict
):
    """Test that an error is raised when a target is provided for N2V."""
    # create configuration
    config = Configuration(**minimum_configuration)
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


def test_train_single_array_no_val(tmp_path: Path, minimum_configuration: dict):
    """Test that CAREamics can be trained with arrays."""
    # training data
    train_array = np.random.rand(32, 32)

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
    assert Path(tmp_path / "checkpoints" / "last.ckpt").exists()

    # export to BMZ
    careamist.export_to_bmz(
        path=tmp_path / "model.zip",
        name="TopModel",
        general_description="A model that just walked in.",
        authors=[{"name": "Amod", "affiliation": "El"}],
    )
    assert (tmp_path / "model.zip").exists()


def test_train_array(tmp_path: Path, minimum_configuration: dict):
    """Test that CAREamics can be trained on arrays."""
    # training data
    train_array = np.random.rand(32, 32)
    val_array = np.random.rand(32, 32)

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
    careamist.train(train_source=train_array, val_source=val_array)

    # check that it trained
    assert Path(tmp_path / "checkpoints" / "last.ckpt").exists()

    # export to BMZ
    careamist.export_to_bmz(
        path=tmp_path / "model.zip",
        name="TopModel",
        general_description="A model that just walked in.",
        authors=[{"name": "Amod", "affiliation": "El"}],
    )
    assert (tmp_path / "model.zip").exists()


@pytest.mark.parametrize("independent_channels", [False, True])
def test_train_array_channel(
    tmp_path: Path, minimum_configuration: dict, independent_channels: bool
):
    """Test that CAREamics can be trained on arrays with channels."""
    # training data
    train_array = np.random.rand(32, 32, 3)
    val_array = np.random.rand(32, 32, 3)

    # create configuration
    config = Configuration(**minimum_configuration)
    config.training_config.num_epochs = 1
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
        path=tmp_path / "model.zip",
        name="TopModel",
        general_description="A model that just walked in.",
        authors=[{"name": "Amod", "affiliation": "El"}],
        channel_names=["red", "green", "blue"],
    )
    assert (tmp_path / "model.zip").exists()


def test_train_array_3d(tmp_path: Path, minimum_configuration: dict):
    """Test that CAREamics can be trained on 3D arrays."""
    # training data
    train_array = np.random.rand(8, 32, 32)
    val_array = np.random.rand(8, 32, 32)

    # create configuration
    minimum_configuration["data_config"]["axes"] = "ZYX"
    minimum_configuration["data_config"]["patch_size"] = (8, 16, 16)
    config = Configuration(**minimum_configuration)
    config.training_config.num_epochs = 1
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
        path=tmp_path / "model.zip",
        name="TopModel",
        general_description="A model that just walked in.",
        authors=[{"name": "Amod", "affiliation": "El"}],
    )
    assert (tmp_path / "model.zip").exists()


def test_train_tiff_files_in_memory_no_val(tmp_path: Path, minimum_configuration: dict):
    """Test that CAREamics can be trained with tiff files in memory."""
    # training data
    train_array = np.random.rand(32, 32)

    # save files
    train_file = tmp_path / "train.tiff"
    tifffile.imwrite(train_file, train_array)

    # create configuration
    config = Configuration(**minimum_configuration)
    config.training_config.num_epochs = 1
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
        path=tmp_path / "model.zip",
        name="TopModel",
        general_description="A model that just walked in.",
        authors=[{"name": "Amod", "affiliation": "El"}],
    )
    assert (tmp_path / "model.zip").exists()


def test_train_tiff_files_in_memory(tmp_path: Path, minimum_configuration: dict):
    """Test that CAREamics can be trained with tiff files in memory."""
    # training data
    train_array = np.random.rand(32, 32)
    val_array = np.random.rand(32, 32)

    # save files
    train_file = tmp_path / "train.tiff"
    tifffile.imwrite(train_file, train_array)

    val_file = tmp_path / "val.tiff"
    tifffile.imwrite(val_file, val_array)

    # create configuration
    config = Configuration(**minimum_configuration)
    config.training_config.num_epochs = 1
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
        path=tmp_path / "model.zip",
        name="TopModel",
        general_description="A model that just walked in.",
        authors=[{"name": "Amod", "affiliation": "El"}],
    )
    assert (tmp_path / "model.zip").exists()


def test_train_tiff_files(tmp_path: Path, minimum_configuration: dict):
    """Test that CAREamics can be trained with tiff files by deactivating
    the in memory dataset.
    """
    # training data
    train_array = np.random.rand(32, 32)
    val_array = np.random.rand(32, 32)

    # save files
    train_file = tmp_path / "train.tiff"
    tifffile.imwrite(train_file, train_array)

    val_file = tmp_path / "val.tiff"
    tifffile.imwrite(val_file, val_array)

    # create configuration
    config = Configuration(**minimum_configuration)
    config.training_config.num_epochs = 1
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
        path=tmp_path / "model.zip",
        name="TopModel",
        general_description="A model that just walked in.",
        authors=[{"name": "Amod", "affiliation": "El"}],
    )
    assert (tmp_path / "model.zip").exists()


def test_train_array_supervised(tmp_path: Path, supervised_configuration: dict):
    """Test that CAREamics can be trained with arrays."""
    # training data
    train_array = np.random.rand(32, 32)
    val_array = np.random.rand(32, 32)
    train_target = np.random.rand(32, 32)
    val_target = np.random.rand(32, 32)

    # create configuration
    config = Configuration(**supervised_configuration)
    config.training_config.num_epochs = 1
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
        path=tmp_path / "model.zip",
        name="TopModel",
        general_description="A model that just walked in.",
        authors=[{"name": "Amod", "affiliation": "El"}],
    )
    assert (tmp_path / "model.zip").exists()


def test_train_tiff_files_in_memory_supervised(
    tmp_path: Path, supervised_configuration: dict
):
    """Test that CAREamics can be trained with tiff files in memory."""
    # training data
    train_array = np.random.rand(32, 32)
    val_array = np.random.rand(32, 32)
    train_target = np.random.rand(32, 32)
    val_target = np.random.rand(32, 32)

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
    config = Configuration(**supervised_configuration)
    config.training_config.num_epochs = 1
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
        path=tmp_path / "model.zip",
        name="TopModel",
        general_description="A model that just walked in.",
        authors=[{"name": "Amod", "affiliation": "El"}],
    )
    assert (tmp_path / "model.zip").exists()


def test_train_tiff_files_supervised(tmp_path: Path, supervised_configuration: dict):
    """Test that CAREamics can be trained with tiff files by deactivating
    the in memory dataset.
    """
    # training data
    train_array = np.random.rand(32, 32)
    val_array = np.random.rand(32, 32)
    train_target = np.random.rand(32, 32)
    val_target = np.random.rand(32, 32)

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
    config = Configuration(**supervised_configuration)
    config.training_config.num_epochs = 1
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
        path=tmp_path / "model.zip",
        name="TopModel",
        general_description="A model that just walked in.",
        authors=[{"name": "Amod", "affiliation": "El"}],
    )
    assert (tmp_path / "model.zip").exists()


@pytest.mark.parametrize("batch_size", [1, 2])
def test_predict_on_array_tiled(
    tmp_path: Path, minimum_configuration: dict, batch_size
):
    """Test that CAREamics can predict on arrays."""
    # training data
    train_array = np.random.rand(32, 32)

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

    # predict CAREamist
    predicted = careamist.predict(
        train_array, batch_size=batch_size, tile_size=(16, 16), tile_overlap=(4, 4)
    )

    assert predicted.squeeze().shape == train_array.shape

    # export to BMZ
    careamist.export_to_bmz(
        path=tmp_path / "model.zip",
        name="TopModel",
        general_description="A model that just walked in.",
        authors=[{"name": "Amod", "affiliation": "El"}],
    )
    assert (tmp_path / "model.zip").exists()


def test_predict_arrays_no_tiling(tmp_path: Path, minimum_configuration: dict):
    """Test that CAREamics can predict on arrays without tiling."""
    # training data
    train_array = np.random.rand(4, 32, 32)

    # create configuration
    config = Configuration(**minimum_configuration)
    config.training_config.num_epochs = 1
    config.data_config.axes = "SYX"
    config.data_config.batch_size = 2
    config.data_config.data_type = SupportedData.ARRAY.value
    config.data_config.patch_size = (8, 8)

    # instantiate CAREamist
    careamist = CAREamist(source=config, work_dir=tmp_path)

    # train CAREamist
    careamist.train(train_source=train_array)

    # predict CAREamist
    predicted = careamist.predict(train_array)

    assert predicted.squeeze().shape == train_array.shape

    # export to BMZ
    careamist.export_to_bmz(
        path=tmp_path / "model.zip",
        name="TopModel",
        general_description="A model that just walked in.",
        authors=[{"name": "Amod", "affiliation": "El"}],
    )
    assert (tmp_path / "model.zip").exists()


@pytest.mark.parametrize("batch_size", [1, 2])
def test_predict_path(tmp_path: Path, minimum_configuration: dict, batch_size):
    """Test that CAREamics can predict with tiff files."""
    # training data
    train_array = np.random.rand(32, 32)

    # save files
    train_file = tmp_path / "train.tiff"
    tifffile.imwrite(train_file, train_array)

    # create configuration
    config = Configuration(**minimum_configuration)
    config.training_config.num_epochs = 1
    config.data_config.axes = "YX"
    config.data_config.batch_size = 2
    config.data_config.data_type = SupportedData.TIFF.value
    config.data_config.patch_size = (8, 8)

    # instantiate CAREamist
    careamist = CAREamist(source=config, work_dir=tmp_path)

    # train CAREamist
    careamist.train(train_source=train_file)

    # predict CAREamist
    predicted = careamist.predict(train_file, batch_size=batch_size)

    # check that it predicted
    assert predicted.squeeze().shape == train_array.shape

    # export to BMZ
    careamist.export_to_bmz(
        path=tmp_path / "model.zip",
        name="TopModel",
        general_description="A model that just walked in.",
        authors=[{"name": "Amod", "affiliation": "El"}],
    )
    assert (tmp_path / "model.zip").exists()


def test_predict_pretrained_checkpoint(tmp_path: Path, pre_trained: Path):
    """Test that CAREamics can be instantiated with a pre-trained network and predict
    on an array."""
    # prediction data
    source_array = np.random.rand(32, 32)

    # instantiate CAREamist
    careamist = CAREamist(source=pre_trained, work_dir=tmp_path)
    assert careamist.cfg.data_config.mean is not None
    assert careamist.cfg.data_config.std is not None

    # predict
    predicted = careamist.predict(source_array)

    # check that it predicted
    assert predicted.squeeze().shape == source_array.shape


def test_predict_pretrained_bmz(tmp_path: Path, pre_trained_bmz: Path):
    """Test that CAREamics can be instantiated with a BMZ archive and predict."""
    # prediction data
    source_array = np.random.rand(32, 32)

    # instantiate CAREamist
    careamist = CAREamist(source=pre_trained_bmz, work_dir=tmp_path)

    # predict
    predicted = careamist.predict(source_array)

    # check that it predicted
    assert predicted.squeeze().shape == source_array.shape


def test_export_bmz_pretrained_prediction(tmp_path: Path, pre_trained: Path):
    """Test that CAREamics can be instantiated with a pre-trained network and exported
    to BMZ after prediction.

    In this case, the careamist extracts the BMZ test data from the prediction
    datamodule.
    """
    # instantiate CAREamist
    careamist = CAREamist(source=pre_trained, work_dir=tmp_path)

    # prediction data
    source_array = np.random.rand(32, 32)
    _ = careamist.predict(source_array)
    assert len(careamist.pred_datamodule.predict_dataloader()) > 0

    # export to BMZ (random array created)
    careamist.export_to_bmz(
        path=tmp_path / "model.zip",
        name="TopModel",
        general_description="A model that just walked in.",
        authors=[{"name": "Amod", "affiliation": "El"}],
    )
    assert (tmp_path / "model.zip").exists()


def test_export_bmz_pretrained_random_array(tmp_path: Path, pre_trained: Path):
    """Test that CAREamics can be instantiated with a pre-trained network and exported
    to BMZ.

    In this case, the careamist creates a random array for the BMZ archive test.
    """
    # instantiate CAREamist
    careamist = CAREamist(source=pre_trained, work_dir=tmp_path)

    # export to BMZ (random array created)
    careamist.export_to_bmz(
        path=tmp_path / "model.zip",
        name="TopModel",
        general_description="A model that just walked in.",
        authors=[{"name": "Amod", "affiliation": "El"}],
    )
    assert (tmp_path / "model.zip").exists()


def test_export_bmz_pretrained_with_array(tmp_path: Path, pre_trained: Path):
    """Test that CAREamics can be instantiated with a pre-trained network and exported
    to BMZ.

    In this case, we provide an array to the BMZ archive test.
    """
    # instantiate CAREamist
    careamist = CAREamist(source=pre_trained, work_dir=tmp_path)

    # alternatively we can pass an array
    array = np.random.rand(32, 32).astype(np.float32)
    careamist.export_to_bmz(
        path=tmp_path / "model2.zip",
        name="TopModel",
        input_array=array[np.newaxis, np.newaxis, ...],
        general_description="A model that just walked in.",
        authors=[{"name": "Amod", "affiliation": "El"}],
    )
    assert (tmp_path / "model2.zip").exists()
