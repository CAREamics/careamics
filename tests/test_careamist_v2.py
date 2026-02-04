from pathlib import Path

import numpy as np
import pytest
import tifffile
from numpy.typing import NDArray

from careamics.careamist_v2 import CAREamistV2
from careamics.config.ng_factories import create_n2v_configuration
from careamics.config.support import SupportedData


def random_array(shape: tuple[int, ...], seed: int = 42) -> NDArray:
    """Return a random array with values between 0 and 255."""
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 255, shape)).astype(np.float32)


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
    config = create_n2v_configuration(
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
    predicted = careamist.predict(train_array, batch_size=batch_size)

    assert len(predicted) == samples
    for i in range(samples):
        assert predicted[i].shape[1:] == (32, 32)


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
    config = create_n2v_configuration(
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
    predicted = careamist.predict(
        train_array, batch_size=batch_size, tile_size=(16, 16), tile_overlap=(4, 4)
    )

    assert len(predicted) == samples
    for i in range(samples):
        assert predicted[i].shape[1:] == (32, 32)


@pytest.mark.mps_gh_fail
@pytest.mark.parametrize("tiled", [True, False])
@pytest.mark.parametrize("n_samples", [1, 2])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_predict_path(
    tmp_path: Path, batch_size: int, n_samples: int, tiled: bool
):
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
    config = create_n2v_configuration(
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
    config = create_n2v_configuration(
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
    predicted = careamist.predict(
        train_array, batch_size=batch_size, tile_size=(16, 16), tile_overlap=(4, 4)
    )

    assert len(predicted) == 1
    assert predicted[0].shape == (3, 32, 32)


@pytest.mark.mps_gh_fail
def test_predict_pretrained_checkpoint(tmp_path: Path, pre_trained: Path):
    """Test that CAREamistV2 can be instantiated with a pre-trained network and predict
    on an array."""
    # prediction data
    source_array = random_array((32, 32))

    # instantiate CAREamist
    careamist = CAREamistV2(checkpoint_path=pre_trained, work_dir=tmp_path)

    # predict
    predicted = careamist.predict(source_array)

    # check that it predicted
    assert len(predicted) == 1
    assert predicted[0].squeeze().shape == source_array.shape


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
    config = create_n2v_configuration(
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
    config = create_n2v_configuration(
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

    for i in range(n_samples):
        assert (tmp_path / "predictions" / f"image_{i}.tiff").is_file()


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
    config = create_n2v_configuration(
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

    for i in range(n_samples):
        assert (tmp_path / "predictions" / f"image_{i}.npy").is_file()


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
    config = create_n2v_configuration(
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

