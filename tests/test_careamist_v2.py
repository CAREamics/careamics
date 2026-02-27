from pathlib import Path

import numpy as np
import pytest
import tifffile
from numpy.typing import NDArray
from pytorch_lightning.callbacks import ModelCheckpoint

from careamics.careamist_v2 import CAREamistV2
from careamics.config.ng_factories import create_advanced_n2v_config


def random_array(shape: tuple[int, ...], seed: int = 42) -> NDArray:
    """Return a random array with values between 0 and 255."""
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 255, shape)).astype(np.float32)


def test_v2_train_error_no_data(tmp_path: Path):
    """Test that a ValueError is raised when no training data is provided."""
    config = create_advanced_n2v_config(
        experiment_name="test",
        data_type="array",
        axes="YX",
        patch_size=(8, 8),
        batch_size=2,
        num_epochs=1,
        roi_size=5,
        masked_pixel_percentage=5,
    )
    careamist = CAREamistV2(config=config, work_dir=tmp_path)

    with pytest.raises(ValueError, match="Training data must be provided"):
        careamist.train()


def test_target_unsupported_warning_n2v(tmp_path: Path):
    """Test that a warning is emitted when a target is provided for N2V."""
    config = create_advanced_n2v_config(
        experiment_name="test",
        data_type="array",
        axes="YX",
        patch_size=(8, 8),
        batch_size=2,
        num_epochs=1,
        roi_size=5,
        masked_pixel_percentage=5,
    )
    careamics = CAREamistV2(config=config, work_dir=tmp_path)

    train_array = np.ones((32, 32))
    val_array = np.ones((32, 32))

    with pytest.warns(match="train_data_target.*ignored"):
        careamics.train(
            train_data=train_array,
            val_data=val_array,
            train_data_target=train_array,
        )


def test_v2_checkpoint_params_propagated(tmp_path: Path):
    """Test that checkpoint callback config is propagated to the ModelCheckpoint."""
    config = create_advanced_n2v_config(
        experiment_name="test",
        data_type="array",
        axes="YX",
        patch_size=(8, 8),
        batch_size=2,
        num_epochs=1,
        roi_size=5,
        masked_pixel_percentage=5,
    )
    config.training_config.checkpoint_callback.save_top_k = 2
    config.training_config.checkpoint_callback.monitor = "val_loss"
    config.training_config.checkpoint_callback.mode = "min"
    config.training_config.checkpoint_callback.save_last = True
    config.training_config.checkpoint_callback.every_n_epochs = 1

    careamist = CAREamistV2(config=config, work_dir=tmp_path)

    ckpt_callbacks = [
        cb for cb in careamist.callbacks if isinstance(cb, ModelCheckpoint)
    ]
    assert len(ckpt_callbacks) == 1
    ckpt_cb = ckpt_callbacks[0]

    assert ckpt_cb.save_top_k == 2
    assert ckpt_cb.monitor == "val_loss"
    assert ckpt_cb.mode == "min"
    assert ckpt_cb.save_last is True
    assert ckpt_cb.every_n_epochs == 1
    assert ckpt_cb.dirpath == str(tmp_path / "checkpoints")


@pytest.mark.mps_gh_fail
def test_v2_train_array(tmp_path: Path):
    """Test that CAREamistV2 can be trained on arrays."""
    train_array = random_array((32, 32), seed=42)
    val_array = random_array((32, 32), seed=43)

    config = create_advanced_n2v_config(
        experiment_name="test",
        data_type="array",
        axes="YX",
        patch_size=(8, 8),
        batch_size=2,
        num_epochs=1,
        roi_size=5,
        masked_pixel_percentage=5,
    )

    careamist = CAREamistV2(config=config, work_dir=tmp_path)
    careamist.train(train_data=train_array, val_data=val_array)

    assert Path(tmp_path / "checkpoints" / "last.ckpt").exists()


@pytest.mark.mps_gh_fail
@pytest.mark.parametrize("independent_channels", [False, True])
def test_v2_train_array_channel(tmp_path: Path, independent_channels: bool):
    """Test that CAREamistV2 can be trained on arrays with channels."""
    train_array = random_array((32, 32, 3), seed=42)
    val_array = random_array((32, 32, 3), seed=43)

    config = create_advanced_n2v_config(
        experiment_name="test",
        data_type="array",
        axes="YXC",
        patch_size=(8, 8),
        batch_size=2,
        num_epochs=1,
        n_channels=3,
        roi_size=5,
        masked_pixel_percentage=5,
    )
    config.algorithm_config.model.in_channels = 3
    config.algorithm_config.model.num_classes = 3
    config.algorithm_config.model.independent_channels = independent_channels

    careamist = CAREamistV2(config=config, work_dir=tmp_path)
    careamist.train(train_data=train_array, val_data=val_array)

    assert Path(tmp_path / "checkpoints" / "last.ckpt").exists()


@pytest.mark.mps_gh_fail
def test_v2_train_array_3d(tmp_path: Path):
    """Test that CAREamistV2 can be trained on 3D arrays."""
    train_array = random_array((8, 32, 32), seed=42)
    val_array = random_array((8, 32, 32), seed=43)

    config = create_advanced_n2v_config(
        experiment_name="test",
        data_type="array",
        axes="ZYX",
        patch_size=(8, 16, 16),
        batch_size=2,
        num_epochs=1,
        roi_size=5,
        masked_pixel_percentage=5,
    )

    careamist = CAREamistV2(config=config, work_dir=tmp_path)
    careamist.train(train_data=train_array, val_data=val_array)

    assert Path(tmp_path / "checkpoints" / "last.ckpt").exists()


@pytest.mark.mps_gh_fail
def test_v2_train_tiff_in_memory(tmp_path: Path):
    """Test that CAREamistV2 can be trained with tiff files in memory."""
    train_array = random_array((32, 32), seed=42)
    val_array = random_array((32, 32), seed=43)

    train_file = tmp_path / "train.tiff"
    tifffile.imwrite(train_file, train_array)

    val_file = tmp_path / "val.tiff"
    tifffile.imwrite(val_file, val_array)

    config = create_advanced_n2v_config(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=(8, 8),
        batch_size=2,
        num_epochs=1,
        roi_size=5,
        masked_pixel_percentage=5,
    )

    careamist = CAREamistV2(config=config, work_dir=tmp_path)
    careamist.train(train_data=train_file, val_data=val_file)

    assert Path(tmp_path / "checkpoints" / "last.ckpt").exists()


@pytest.mark.mps_gh_fail
def test_v2_train_tiff_not_in_memory(tmp_path: Path):
    """Test N2V training from tiff files with in_memory=False (on-disk reading)."""
    train_array = random_array((32, 32), seed=42)
    val_array = random_array((32, 32), seed=43)

    train_file = tmp_path / "train.tiff"
    tifffile.imwrite(train_file, train_array)

    val_file = tmp_path / "val.tiff"
    tifffile.imwrite(val_file, val_array)

    config = create_advanced_n2v_config(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=(8, 8),
        batch_size=2,
        num_epochs=1,
        roi_size=5,
        masked_pixel_percentage=5,
        in_memory=False,
    )

    careamist = CAREamistV2(config=config, work_dir=tmp_path)
    careamist.train(train_data=train_file, val_data=val_file)

    assert Path(tmp_path / "checkpoints" / "last.ckpt").exists()


@pytest.mark.skip(reason="CAREamistV2.train() is not yet implemented")
@pytest.mark.mps_gh_fail
@pytest.mark.parametrize("samples", [1, 2, 4])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_predict_arrays_no_tiling(tmp_path: Path, batch_size: int, samples: int):
    """Test that CAREamistV2 can predict on arrays without tiling."""
    # training data
    train_array = random_array((samples, 32, 32))

    # create configuration
    patch_size = [8, 8]
    masked_pixel_percentage = 100 / np.prod(patch_size)
    config = create_advanced_n2v_config(
        experiment_name="test",
        data_type="array",
        axes="SYX",
        patch_size=patch_size,
        batch_size=2,
        masked_pixel_percentage=masked_pixel_percentage,
    )

    # instantiate CAREamist
    careamist = CAREamistV2(config=config, work_dir=tmp_path)

    # train CAREamist
    careamist.train(train_data=train_array)

    # predict CAREamist
    predicted, _ = careamist.predict(train_array, batch_size=batch_size)

    assert predicted[0].shape == (samples, 32, 32)


@pytest.mark.skip(reason="CAREamistV2.train() is not yet implemented")
@pytest.mark.mps_gh_fail
@pytest.mark.parametrize("samples", [1, 2, 4])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_predict_on_array_tiled(tmp_path: Path, batch_size: int, samples: int):
    """Test that CAREamistV2 can predict on arrays with tiling."""
    # training data
    train_array = random_array((samples, 32, 32))

    # create configuration
    patch_size = [8, 8]
    masked_pixel_percentage = 100 / np.prod(patch_size)
    config = create_advanced_n2v_config(
        experiment_name="test",
        data_type="array",
        axes="SYX",
        patch_size=patch_size,
        batch_size=2,
        masked_pixel_percentage=masked_pixel_percentage,
    )

    # instantiate CAREamist
    careamist = CAREamistV2(config=config, work_dir=tmp_path)

    # train CAREamist
    careamist.train(train_data=train_array)

    # predict CAREamist
    predicted, _ = careamist.predict(
        train_array, batch_size=batch_size, tile_size=(16, 16), tile_overlap=(4, 4)
    )

    # TODO revisit after fixing prediction reshaping to orig shape
    assert predicted[0].shape[0] == samples
    assert predicted[0].shape[-2:] == (32, 32)


@pytest.mark.skip(reason="CAREamistV2.train() is not yet implemented")
@pytest.mark.mps_gh_fail
@pytest.mark.parametrize("tiled", [True, False])
@pytest.mark.parametrize("n_samples", [1, 2])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_predict_path(tmp_path: Path, batch_size: int, n_samples: int, tiled: bool):
    """Test that CAREamistV2 can predict with tiff files."""
    # training data
    train_array = random_array((32, 32))

    # save files
    for i in range(n_samples):
        train_file = tmp_path / f"train_{i}.tiff"
        tifffile.imwrite(train_file, train_array)

    # create configuration
    patch_size = [8, 8]
    masked_pixel_percentage = 100 / np.prod(patch_size)
    config = create_advanced_n2v_config(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=patch_size,
        batch_size=2,
        masked_pixel_percentage=masked_pixel_percentage,
    )

    # instantiate CAREamist
    careamist = CAREamistV2(config=config, work_dir=tmp_path)

    # train CAREamist
    train_file = tmp_path / "train_0.tiff"
    careamist.train(train_data=train_file)

    if tiled:
        tile_size = (16, 16)
        tile_overlap = (4, 4)
    else:
        tile_size = None
        tile_overlap = None

    # predict CAREamist
    predicted, _ = careamist.predict(
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


@pytest.mark.skip(reason="CAREamistV2.train() is not yet implemented")
@pytest.mark.mps_gh_fail
@pytest.mark.parametrize("independent_channels", [False, True])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_predict_tiled_channel(
    tmp_path: Path,
    independent_channels: bool,
    batch_size: int,
):
    """Test that CAREamistV2 can be trained on arrays with channels."""
    # training data
    train_array = random_array((3, 32, 32))
    val_array = random_array((3, 32, 32))

    # create configuration
    patch_size = [8, 8]
    masked_pixel_percentage = 100 / np.prod(patch_size)
    config = create_advanced_n2v_config(
        experiment_name="test",
        data_type="array",
        axes="CYX",
        patch_size=patch_size,
        batch_size=2,
        n_channels=3,
        independent_channels=independent_channels,
        masked_pixel_percentage=masked_pixel_percentage,
    )

    # instantiate CAREamist
    careamist = CAREamistV2(config=config, work_dir=tmp_path)

    # train CAREamist
    careamist.train(train_data=train_array, val_data=val_array)

    # predict CAREamist
    predicted, _ = careamist.predict(
        train_array, batch_size=batch_size, tile_size=(16, 16), tile_overlap=(4, 4)
    )

    assert len(predicted) == 1
    assert predicted[0].squeeze().shape == (3, 32, 32)


@pytest.mark.skip(reason="CAREamistV2.train() is not yet implemented")
@pytest.mark.mps_gh_fail
def test_predict_to_disk_path_tiff(tmp_path: Path):
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
    patch_size = [8, 8]
    masked_pixel_percentage = 100 / np.prod(patch_size)
    config = create_advanced_n2v_config(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=patch_size,
        batch_size=2,
        masked_pixel_percentage=masked_pixel_percentage,
    )

    # train
    careamist = CAREamistV2(config=config, work_dir=tmp_path)
    careamist.train(train_data=image_dir)

    # predict to disk
    careamist.predict_to_disk(pred_data=image_dir)

    for i in range(n_samples):
        assert (tmp_path / "predictions" / f"image_{i}.tiff").is_file()


@pytest.mark.skip(reason="CAREamistV2.train() is not yet implemented")
@pytest.mark.mps_gh_fail
def test_predict_to_disk_datamodule_tiff(tmp_path: Path):
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
    patch_size = [8, 8]
    masked_pixel_percentage = 100 / np.prod(patch_size)
    config = create_advanced_n2v_config(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=patch_size,
        batch_size=2,
        masked_pixel_percentage=masked_pixel_percentage,
    )

    # train
    careamist = CAREamistV2(config=config, work_dir=tmp_path)
    careamist.train(train_data=image_dir)

    # create datamodule
    from careamics.lightning.dataset_ng.data_module import CareamicsDataModule

    pred_data_config = config.data_config.convert_mode("predicting")
    datamodule = CareamicsDataModule(
        data_config=pred_data_config,
        pred_data=image_dir,
    )

    # predict to disk
    careamist.predict_to_disk(pred_data=datamodule)

    # Check if predictions directory exists and list files
    predictions_dir = tmp_path / "predictions"
    if predictions_dir.exists():
        created_files = list(predictions_dir.rglob("*"))
        # Files may be in subdirectories matching source structure
        for i in range(n_samples):
            # Try both flat and nested structure
            expected_files = [
                predictions_dir / f"image_{i}.tiff",
                predictions_dir / "images" / f"image_{i}.tiff",
            ]
            assert any(
                f.is_file() for f in expected_files
            ), f"Expected file for sample {i} not found. Created files: {created_files}"
    else:
        # Fallback: check in work_dir directly
        for i in range(n_samples):
            assert (tmp_path / "predictions" / f"image_{i}.tiff").is_file()


@pytest.mark.skip(reason="CAREamistV2.train() is not yet implemented")
@pytest.mark.mps_gh_fail
def test_predict_to_disk_custom(tmp_path: Path):
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
    patch_size = [8, 8]
    masked_pixel_percentage = 100 / np.prod(patch_size)
    config = create_advanced_n2v_config(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=patch_size,
        batch_size=2,
        masked_pixel_percentage=masked_pixel_percentage,
    )

    # train
    careamist = CAREamistV2(config=config, work_dir=tmp_path)
    careamist.train(train_data=image_dir)

    # predict to disk
    careamist.predict_to_disk(
        pred_data=image_dir,
        write_type="custom",
        write_extension=".npy",
        write_func=write_numpy,
    )

    # Check if predictions directory exists and list files
    predictions_dir = tmp_path / "predictions"
    if predictions_dir.exists():
        created_files = list(predictions_dir.rglob("*"))
        # Files may be in subdirectories matching source structure
        for i in range(n_samples):
            # Try both flat and nested structure
            expected_files = [
                predictions_dir / f"image_{i}.npy",
                predictions_dir / "images" / f"image_{i}.npy",
            ]
            assert any(
                f.is_file() for f in expected_files
            ), f"Expected file for sample {i} not found. Created files: {created_files}"
    else:
        # Fallback: check in work_dir directly
        for i in range(n_samples):
            assert (tmp_path / "predictions" / f"image_{i}.npy").is_file()


@pytest.mark.skip(reason="CAREamistV2.train() is not yet implemented")
@pytest.mark.mps_gh_fail
def test_predict_to_disk_custom_raises(tmp_path: Path):
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
    patch_size = [8, 8]
    masked_pixel_percentage = 100 / np.prod(patch_size)
    config = create_advanced_n2v_config(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=patch_size,
        batch_size=2,
        masked_pixel_percentage=masked_pixel_percentage,
    )

    # train
    careamist = CAREamistV2(config=config, work_dir=tmp_path)
    careamist.train(train_data=image_dir)

    with pytest.raises(ValueError):
        # no write extension provided
        careamist.predict_to_disk(
            pred_data=image_dir,
            write_type="custom",
            write_extension=None,
            write_func=write_numpy,
        )
    with pytest.raises(ValueError):
        # no write func provided.
        careamist.predict_to_disk(
            pred_data=image_dir,
            write_type="custom",
            write_extension=".npy",
            write_func=None,
        )
