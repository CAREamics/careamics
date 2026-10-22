from pathlib import Path

import numpy as np
import pytest
import zarr
from tifffile import imwrite
from yaozarrs import validate_zarr_store

from careamics import CAREamist
from careamics.config import create_advanced_n2v_config, create_care_config
from tests.conftest import (
    create_image_collection_ome_zarr,
    create_single_image_ome_zarr,
)


def _create_plain_root_array_zarr(store_path: Path, data: np.ndarray) -> str:
    zarr.open_array(
        store_path,
        mode="w",
        shape=data.shape,
        dtype=data.dtype,
        chunks=(16, 16),
    )[:] = data
    return store_path.as_uri()


def _train_n2v_model(
    tmp_path: Path,
    train_data: list[str],
    val_data: list[str],
    zarr_backend: str = "zarr",
    data_type: str = "zarr",
) -> CAREamist:
    cfg = create_advanced_n2v_config(
        experiment_name="n2v_zarr",
        data_type=data_type,
        axes="YX",
        patch_size=(32, 32),
        batch_size=2,
        num_epochs=2,
        zarr_backend=zarr_backend,
    )
    careamist = CAREamist(cfg, work_dir=tmp_path)
    careamist.train(train_data=train_data, val_data=val_data)
    return careamist


def _train_care_model(
    tmp_path: Path,
    train_data: list[str],
    train_data_target: list[str],
    val_data: list[str],
    val_data_target: list[str],
    data_type: str = "zarr",
) -> CAREamist:
    cfg = create_care_config(
        experiment_name="n2v_zarr",
        data_type=data_type,
        axes="YX",
        patch_size=(16, 16),
        batch_size=2,
        num_epochs=2,
    )
    careamist = CAREamist(cfg, work_dir=tmp_path)
    careamist.train(
        train_data=train_data,
        train_data_target=train_data_target,
        val_data=val_data,
        val_data_target=val_data_target,
    )
    return careamist


def _assert_ome_metadata_at_scale(
    source_group: zarr.Group,
    prediction_group: zarr.Group,
    scale: str = "0",
) -> None:
    """Assert conserved OME-NGFF metadata for one multiscale level."""
    source_ome = source_group.attrs["ome"]
    prediction_ome = prediction_group.attrs["ome"]
    source_multiscale = source_ome["multiscales"][0]
    prediction_multiscale = prediction_ome["multiscales"][0]

    assert prediction_ome["version"] == source_ome["version"]
    assert prediction_multiscale["axes"] == source_multiscale["axes"]

    source_dataset = next(
        dataset for dataset in source_multiscale["datasets"] if dataset["path"] == scale
    )
    prediction_dataset = next(
        dataset
        for dataset in prediction_multiscale["datasets"]
        if dataset["path"] == scale
    )
    assert (
        prediction_dataset["coordinateTransformations"]
        == source_dataset["coordinateTransformations"]
    )


@pytest.mark.mps_gh_fail
def test_smoke_n2v_plain_zarr_root_array(tmp_path: Path) -> None:
    """Test that a plain zarr root array gets written as an Image OME-Zarr."""
    rng = np.random.default_rng(42)
    train_data = rng.integers(0, 255, (32, 32)).astype(np.float32)
    val_data = rng.integers(0, 255, (32, 32)).astype(np.float32)
    pred_data = rng.integers(0, 255, (32, 32)).astype(np.float32)

    train_uri = _create_plain_root_array_zarr(tmp_path / "train_image.zarr", train_data)
    val_uri = _create_plain_root_array_zarr(tmp_path / "val_image.zarr", val_data)
    pred_uri = _create_plain_root_array_zarr(tmp_path / "input_image.zarr", pred_data)

    careamist = _train_n2v_model(
        tmp_path,
        train_data=[train_uri],
        val_data=[val_uri],
    )

    # tiled prediction to disk
    careamist.predict_to_disk(
        pred_data=[pred_uri],
        prediction_dir="predictions",
        tile_size=(16, 16),
        tile_overlap=(4, 4),
    )
    output_store = tmp_path / "predictions" / "input_image_output.zarr"

    assert output_store.exists()
    validate_zarr_store(output_store)

    # whole image prediction to disk
    careamist.predict_to_disk(
        pred_data=[pred_uri],
        prediction_dir="whole_image_predictions",
    )
    whole_image_output_store = (
        tmp_path / "whole_image_predictions" / "input_image_output.zarr"
    )

    assert whole_image_output_store.exists()
    validate_zarr_store(whole_image_output_store)


@pytest.mark.mps_gh_fail
@pytest.mark.parametrize("backend", ["zarr", "zarrs", "tensorstore"])
def test_smoke_n2v_single_image_ome_zarr(tmp_path: Path, backend: str) -> None:
    """Test that an OME-NGFF single array gets written as an Image OME-Zarr.

    Note: this test also run the various zarr backends.
    """
    rng = np.random.default_rng(42)
    train_array = rng.integers(0, 255, (32, 32)).astype(np.float32)
    val_array = rng.integers(0, 255, (32, 32)).astype(np.float32)
    pred_array = rng.integers(0, 255, (32, 32)).astype(np.float32)
    spatial_scale = (0.25, 0.75)

    train_uri = create_single_image_ome_zarr(
        tmp_path / "train_image.zarr",
        train_array,
        chunks=(16, 16),
        spatial_scale=spatial_scale,
    ).as_uri()
    val_uri = create_single_image_ome_zarr(
        tmp_path / "val_image.zarr",
        val_array,
        chunks=(16, 16),
        spatial_scale=spatial_scale,
    ).as_uri()
    pred_uri = create_single_image_ome_zarr(
        tmp_path / "input_image.zarr",
        pred_array,
        chunks=(16, 16),
        spatial_scale=spatial_scale,
    ).as_uri()

    careamist = _train_n2v_model(
        tmp_path, train_data=[train_uri], val_data=[val_uri], zarr_backend=backend
    )
    careamist.predict_to_disk(
        pred_data=[pred_uri],
        prediction_dir="predictions",
        tile_size=(16, 16),
        tile_overlap=(4, 4),
    )
    output_store = tmp_path / "predictions" / "input_image_output.zarr"

    assert output_store.exists()
    validate_zarr_store(output_store)
    _assert_ome_metadata_at_scale(
        zarr.open_group(tmp_path / "input_image.zarr", mode="r"),
        zarr.open_group(output_store, mode="r"),
    )

    # whole image prediction to disk
    careamist.predict_to_disk(
        pred_data=[pred_uri],
        prediction_dir="whole_image_predictions",
    )
    whole_image_output_store = (
        tmp_path / "whole_image_predictions" / "input_image_output.zarr"
    )

    assert whole_image_output_store.exists()
    validate_zarr_store(whole_image_output_store)


@pytest.mark.mps_gh_fail
def test_smoke_n2v_collection_ome_zarr(tmp_path: Path) -> None:
    """Test that an OME-NGFF image collection is written as an OME-Zarr collection."""
    rng = np.random.default_rng(42)
    train_images = {
        "img_0": rng.integers(0, 255, (32, 32)).astype(np.float32),
        "img_1": rng.integers(0, 255, (32, 32)).astype(np.float32),
    }
    val_images = {
        "img_0": rng.integers(0, 255, (32, 32)).astype(np.float32),
        "img_1": rng.integers(0, 255, (32, 32)).astype(np.float32),
    }
    pred_images = {
        "img_0": rng.integers(0, 255, (32, 32)).astype(np.float32),
        "img_1": rng.integers(0, 255, (32, 32)).astype(np.float32),
    }
    spatial_scale = (0.25, 0.75)

    train_uri = create_image_collection_ome_zarr(
        tmp_path / "train_collection.zarr",
        train_images,
        chunks=(16, 16),
        spatial_scale=spatial_scale,
    ).as_uri()
    val_uri = create_image_collection_ome_zarr(
        tmp_path / "val_collection.zarr",
        val_images,
        chunks=(16, 16),
        spatial_scale=spatial_scale,
    ).as_uri()
    pred_uri = create_image_collection_ome_zarr(
        tmp_path / "input_collection.zarr",
        pred_images,
        chunks=(16, 16),
        spatial_scale=spatial_scale,
    ).as_uri()

    careamist = _train_n2v_model(
        tmp_path,
        train_data=[train_uri],
        val_data=[val_uri],
    )
    careamist.predict_to_disk(
        pred_data=[pred_uri],
        prediction_dir="predictions",
        tile_size=(16, 16),
        tile_overlap=(4, 4),
    )
    output_store = tmp_path / "predictions" / "input_collection_output.zarr"
    assert output_store.exists()
    validate_zarr_store(output_store)
    source_group = zarr.open_group(tmp_path / "input_collection.zarr", mode="r")
    output_group = zarr.open_group(output_store, mode="r")
    for image_name in pred_images:
        _assert_ome_metadata_at_scale(
            source_group[image_name],
            output_group[image_name],
        )

    # whole image prediction to disk
    careamist.predict_to_disk(
        pred_data=[pred_uri],
        prediction_dir="whole_image_predictions",
    )
    whole_image_output_store = (
        tmp_path / "whole_image_predictions" / "input_collection_output.zarr"
    )

    assert whole_image_output_store.exists()
    validate_zarr_store(whole_image_output_store)


@pytest.mark.mps_gh_fail
def test_smoke_care_supervised_same_ome_zarr(tmp_path: Path) -> None:
    """Test that supervised CARE prediction writes a valid OME-Zarr collection."""
    rng = np.random.default_rng(42)

    train_input = rng.integers(0, 255, (32, 32)).astype(np.float32)
    train_target = rng.integers(0, 255, (32, 32)).astype(np.float32)
    val_input = rng.integers(0, 255, (32, 32)).astype(np.float32)
    val_target = rng.integers(0, 255, (32, 32)).astype(np.float32)
    pred_input = rng.integers(0, 255, (32, 32)).astype(np.float32)

    train_store = tmp_path / "train_supervised.zarr"
    create_image_collection_ome_zarr(
        train_store,
        {"input": train_input, "target": train_target},
        chunks=(16, 16),
    )
    train_store_uri = train_store.as_uri()
    train_input_uri = f"{train_store_uri}/input"
    train_target_uri = f"{train_store_uri}/target"

    val_store = tmp_path / "val_supervised.zarr"
    create_image_collection_ome_zarr(
        val_store,
        {"input": val_input, "target": val_target},
        chunks=(16, 16),
    )
    val_store_uri = val_store.as_uri()
    val_input_uri = f"{val_store_uri}/input"
    val_target_uri = f"{val_store_uri}/target"

    pred_store = tmp_path / "pred_supervised.zarr"
    create_image_collection_ome_zarr(
        pred_store,
        {"input": pred_input, "target": train_target},
        chunks=(16, 16),
    )
    pred_input_uri = f"{pred_store.as_uri()}/input"

    careamist = _train_care_model(
        tmp_path,
        train_data=[train_input_uri],
        train_data_target=[train_target_uri],
        val_data=[val_input_uri],
        val_data_target=[val_target_uri],
    )
    careamist.predict_to_disk(
        pred_data=[pred_input_uri],
        prediction_dir="predictions",
        tile_size=(16, 16),
        tile_overlap=(4, 4),
    )
    output_store = tmp_path / "predictions" / "pred_supervised_output.zarr"
    assert output_store.exists()
    validate_zarr_store(output_store)

    # whole image prediction to disk
    careamist.predict_to_disk(
        pred_data=[pred_input_uri],
        prediction_dir="whole_image_predictions",
    )
    whole_image_output_store = (
        tmp_path / "whole_image_predictions" / "pred_supervised_output.zarr"
    )

    assert whole_image_output_store.exists()
    validate_zarr_store(whole_image_output_store)


@pytest.mark.mps_gh_fail
def test_smoke_n2v_arrays_to_zarr(tmp_path: Path) -> None:
    """Test that arrays are written as an OME-Zarr collection."""
    rng = np.random.default_rng(42)
    train_images = [rng.integers(0, 255, (32, 32)).astype(np.float32) for _ in range(2)]

    careamist = _train_n2v_model(
        tmp_path, train_data=train_images, val_data=train_images, data_type="array"
    )
    careamist.predict_to_disk(
        pred_data=train_images,
        prediction_dir="predictions",
        tile_size=(16, 16),
        tile_overlap=(4, 4),
        write_type="zarr",
    )
    output_store = tmp_path / "predictions" / "prediction.zarr"
    assert output_store.exists()
    validate_zarr_store(output_store)

    # count number of arrays in the store
    group = zarr.open_group(output_store)
    assert "OME" in group  # series
    assert len(list(group["0"].arrays())) == 1
    assert len(list(group["1"].arrays())) == 1


@pytest.mark.mps_gh_fail
def test_smoke_n2v_tiff_to_zarr(tmp_path: Path) -> None:
    """Test that an tiff files are written as several OME-Zarr."""
    rng = np.random.default_rng(42)
    train_images = [rng.integers(0, 255, (32, 32)).astype(np.float32) for _ in range(2)]

    train_path = tmp_path / "train"
    train_path.mkdir()

    for i, arr in enumerate(train_images):
        imwrite(train_path / f"image_{i}.tif", arr)

    careamist = _train_n2v_model(
        tmp_path, train_data=train_path, val_data=train_path, data_type="tiff"
    )
    careamist.predict_to_disk(
        pred_data=train_path,
        prediction_dir="predictions",
        tile_size=(16, 16),
        tile_overlap=(4, 4),
        write_type="zarr",
    )

    output_store = tmp_path / "predictions" / "prediction.zarr"
    assert output_store.exists()
    validate_zarr_store(output_store)

    # count number of arrays in the store
    group = zarr.open_group(output_store)
    assert "OME" in group  # series
    for i in range(len(train_images)):
        assert len(list(group[f"image_{i}"].arrays())) == 1, f"Failed for image_{i}"
