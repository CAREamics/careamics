from pathlib import Path

import numpy as np
import pytest
import tifffile
import zarr
from numpy.typing import NDArray

from careamics.careamist import CAREamist
from careamics.config.factories import create_advanced_n2v_config


def random_array(shape: tuple[int, ...], seed: int = 42) -> NDArray:
    """Return a random array with values between 0 and 255."""
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 255, shape)).astype(np.float32)


@pytest.mark.mps_gh_fail
@pytest.mark.parametrize("samples", [1, 2, 4])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_predict_arrays_no_tiling(tmp_path: Path, batch_size: int, samples: int):
    """Test that CAREamist can predict on arrays without tiling."""
    train_array = random_array((samples, 32, 32), seed=42)
    val_array = random_array((samples, 32, 32), seed=43)

    config = create_advanced_n2v_config(
        experiment_name="test",
        data_type="array",
        axes="SYX",
        patch_size=(8, 8),
        batch_size=2,
        num_epochs=1,
        roi_size=5,
        masked_pixel_percentage=5,
    )

    careamist = CAREamist(config=config, work_dir=tmp_path)
    careamist.train(train_data=train_array, val_data=val_array)

    predicted, _ = careamist.predict(train_array, batch_size=batch_size)

    assert predicted[0].shape == (samples, 32, 32)


@pytest.mark.mps_gh_fail
@pytest.mark.parametrize("samples", [1, 2, 4])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_predict_on_array_tiled(tmp_path: Path, batch_size: int, samples: int):
    """Test that CAREamist can predict on arrays with tiling."""
    train_array = random_array((samples, 32, 32), seed=42)
    val_array = random_array((samples, 32, 32), seed=43)

    config = create_advanced_n2v_config(
        experiment_name="test",
        data_type="array",
        axes="SYX",
        patch_size=(8, 8),
        batch_size=2,
        num_epochs=1,
        roi_size=5,
        masked_pixel_percentage=5,
    )

    careamist = CAREamist(config=config, work_dir=tmp_path)
    careamist.train(train_data=train_array, val_data=val_array)

    predicted, _ = careamist.predict(
        train_array, batch_size=batch_size, tile_size=(16, 16), tile_overlap=(4, 4)
    )

    assert predicted[0].shape == (samples, 32, 32)


@pytest.mark.mps_gh_fail
@pytest.mark.parametrize("tiled", [True, False])
def test_predict_path(tmp_path: Path, tiled: bool):
    """Test that CAREamist can predict with tiff files."""
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

    careamist = CAREamist(config=config, work_dir=tmp_path)
    careamist.train(train_data=train_file, val_data=val_file)

    if tiled:
        tile_size = (16, 16)
        tile_overlap = (4, 4)
    else:
        tile_size = None
        tile_overlap = None

    predicted, _ = careamist.predict(
        train_file,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
    )

    for p in predicted:
        assert p.squeeze().shape == train_array.shape


@pytest.mark.mps_gh_fail
@pytest.mark.parametrize("independent_channels", [False, True])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_predict_tiled_channel(
    tmp_path: Path,
    independent_channels: bool,
    batch_size: int,
):
    """Test that CAREamist can predict on arrays with channels and tiling."""
    train_array = random_array((3, 32, 32))
    val_array = random_array((3, 32, 32))

    config = create_advanced_n2v_config(
        experiment_name="test",
        data_type="array",
        axes="CYX",
        patch_size=(8, 8),
        batch_size=2,
        num_epochs=1,
        n_channels=3,
        independent_channels=independent_channels,
        roi_size=5,
        masked_pixel_percentage=5,
    )

    careamist = CAREamist(config=config, work_dir=tmp_path)
    careamist.train(train_data=train_array, val_data=val_array)

    predicted, _ = careamist.predict(
        train_array, batch_size=batch_size, tile_size=(16, 16), tile_overlap=(4, 4)
    )

    assert len(predicted) == 1
    assert predicted[0].squeeze().shape == (3, 32, 32)


@pytest.mark.skip(reason="TODO revisit after fixing stats in convert mode")
@pytest.mark.mps_gh_fail
def test_predict_channel_subset(tmp_path: Path):
    """Test that CAREamist can predict on a subset of channels."""
    train_array = random_array((3, 32, 32), seed=42)
    val_array = random_array((3, 32, 32), seed=43)

    config = create_advanced_n2v_config(
        experiment_name="test",
        data_type="array",
        axes="CYX",
        patch_size=(8, 8),
        batch_size=2,
        num_epochs=1,
        n_channels=3,
        roi_size=5,
        masked_pixel_percentage=5,
    )

    careamist = CAREamist(config=config, work_dir=tmp_path)
    careamist.train(train_data=train_array, val_data=val_array)

    predicted, _ = careamist.predict(
        train_array, channels=[0, 2], tile_size=(16, 16), tile_overlap=(4, 4)
    )

    assert len(predicted) == 1
    assert predicted[0].squeeze().shape == (2, 32, 32)


@pytest.mark.mps_gh_fail
def test_predict_to_disk_path_tiff(tmp_path: Path):
    """Test predict_to_disk with path source and tiff write type."""
    train_array = random_array((32, 32), seed=42)
    val_array = random_array((32, 32), seed=43)

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    n_samples = 2
    for i in range(n_samples):
        tifffile.imwrite(image_dir / f"image_{i}.tiff", train_array)

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

    careamist = CAREamist(config=config, work_dir=tmp_path)
    careamist.train(train_data=image_dir, val_data=val_file)

    careamist.predict_to_disk(pred_data=image_dir)

    for i in range(n_samples):
        assert (tmp_path / "predictions" / f"image_{i}.tiff").is_file()


@pytest.mark.mps_gh_fail
def test_predict_to_disk_custom(tmp_path: Path):
    """Test predict_to_disk with custom write type."""

    def write_numpy(file_path: Path, img: NDArray, *args, **kwargs) -> None:
        np.save(file=file_path, arr=img)

    train_array = random_array((32, 32), seed=42)
    val_array = random_array((32, 32), seed=43)

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    n_samples = 2
    for i in range(n_samples):
        tifffile.imwrite(image_dir / f"image_{i}.tiff", train_array)

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

    careamist = CAREamist(config=config, work_dir=tmp_path)
    careamist.train(train_data=image_dir, val_data=val_file)

    careamist.predict_to_disk(
        pred_data=image_dir,
        write_type="custom",
        write_extension=".npy",
        write_func=write_numpy,
    )

    predictions_dir = tmp_path / "predictions"
    for i in range(n_samples):
        assert (predictions_dir / f"image_{i}.npy").is_file()


def test_predict_to_disk_custom_raises(tmp_path: Path):
    """Test predict_to_disk raises ValueError for incomplete custom write config."""

    def write_numpy(file_path: Path, img: NDArray, *args, **kwargs) -> None:
        np.save(file=file_path, arr=img)

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

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    tifffile.imwrite(image_dir / "image.tiff", random_array((32, 32)))

    careamist = CAREamist(config=config, work_dir=tmp_path)

    with pytest.raises(ValueError, match="write_extension"):
        careamist.predict_to_disk(
            pred_data=image_dir,
            write_type="custom",
            write_extension=None,
            write_func=write_numpy,
        )
    with pytest.raises(ValueError, match="write_func"):
        careamist.predict_to_disk(
            pred_data=image_dir,
            write_type="custom",
            write_extension=".npy",
            write_func=None,
        )


def test_predict_invalid_spatial_dims_no_tiling_raises(tmp_path: Path):
    """Test that predict raises ValueError for invalid spatial dims without tiling."""
    config = create_advanced_n2v_config(
        experiment_name="test",
        data_type="array",
        axes="YX",
        patch_size=(8, 8),
        batch_size=1,
        num_epochs=1,
        roi_size=5,
        masked_pixel_percentage=5,
    )

    careamist = CAREamist(config=config, work_dir=tmp_path)

    pred_array = random_array((15, 30))

    with pytest.raises(ValueError, match="tiling"):
        careamist.predict(pred_data=pred_array)

    with pytest.raises(ValueError, match="tiling"):
        careamist.predict_to_disk(pred_data=pred_array)


def _write_zarr_input(path: Path, array: NDArray) -> str:
    """Write array to a zarr store and return the store path of the array."""
    group = zarr.open_group(path, mode="w")
    arr = group.create_array("array", data=array, chunks="auto")
    return str(arr.store_path)


def _train_in_memory_tiff_careamist(tmp_path: Path) -> CAREamist:
    """Train an N2V CAREamist on in-memory tiff data (in_memory defaults to True)."""
    train_array = random_array((32, 32), seed=42)
    val_array = random_array((32, 32), seed=43)

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    tifffile.imwrite(image_dir / "image.tiff", train_array)

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
    # training on tiff defaults `in_memory` to True
    assert config.data_config.in_memory is True

    careamist = CAREamist(config=config, work_dir=tmp_path)
    careamist.train(train_data=image_dir, val_data=val_file)
    return careamist


@pytest.mark.mps_gh_fail
def test_predict_to_disk_zarr_input_defaults_to_zarr(tmp_path: Path):
    """Predicting on zarr input defaults the write type to zarr.

    Also checks that a stale in-memory training value does not carry over and
    conflict with the zarr prediction data type.
    """
    careamist = _train_in_memory_tiff_careamist(tmp_path)

    zarr_path = _write_zarr_input(tmp_path / "pred.zarr", random_array((32, 32)))

    # write_type and in_memory are intentionally omitted
    careamist.predict_to_disk(
        pred_data=[zarr_path],
        data_type="zarr",
        tile_size=(16, 16),
        tile_overlap=(4, 4),
    )

    outputs = list((tmp_path / "predictions").glob("*.zarr"))
    assert len(outputs) == 1


def test_predict_to_disk_zarr_explicit_in_memory_raises(tmp_path: Path):
    """An explicit `in_memory=True` with zarr data still raises."""
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
    careamist = CAREamist(config=config, work_dir=tmp_path)

    zarr_path = _write_zarr_input(tmp_path / "pred.zarr", random_array((32, 32)))

    with pytest.raises(ValueError, match="in_memory"):
        careamist.predict_to_disk(
            pred_data=[zarr_path],
            data_type="zarr",
            tile_size=(16, 16),
            tile_overlap=(4, 4),
            in_memory=True,
        )
