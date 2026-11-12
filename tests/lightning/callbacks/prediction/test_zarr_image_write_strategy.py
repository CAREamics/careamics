"""Tests for whole-image Zarr prediction writing."""

from pathlib import Path

import numpy as np
import pytest
import zarr
from yaozarrs import validate_zarr_store

from careamics.dataset.image_region_data import ImageRegionData
from careamics.dataset.image_stack.zarr_access import (
    DefaultZarrAccess,
    path_to_file_uri,
)
from careamics.lightning.callbacks.prediction import ZarrImageWriteStrategy

BACKENDS = [
    pytest.param(DefaultZarrAccess, id="zarr"),
]


def _image_regions(
    source: str,
    additional_metadata: dict,
    data_idx: int = 0,
) -> list[ImageRegionData]:
    """Create samples belonging to one image."""
    return [
        ImageRegionData(
            data=np.full((2, 4, 8, 8), sample_idx, dtype=np.float32),  # no S
            source=source,
            data_shape=(3, 2, 4, 8, 8),
            dtype="float32",
            axes="SCZYX",
            target_axes="SCZYX",
            original_data_shape=(3, 2, 4, 8, 8),
            region_spec={
                "data_idx": data_idx,
                "sample_idx": sample_idx,
                "coords": (0, 0, 0),
                "patch_size": (8, 8, 8),
            },
            additional_metadata=additional_metadata,
        )
        for sample_idx in range(3)
    ]


@pytest.mark.parametrize("access_cls", BACKENDS)
@pytest.mark.parametrize("source_kind", ["array", "file", "zarr"])
def test_write_whole_image(tmp_path: Path, access_cls, source_kind: str) -> None:
    """Test whole-image output routing and OME-Zarr validity."""
    if source_kind == "array":
        source = "array"
        output_store = tmp_path / "array_0.ome.zarr"
        array_path = "0"
        additional_metadata = {}
    elif source_kind == "file":
        source = str(tmp_path / "input.tiff")
        output_store = tmp_path / "input.ome.zarr"
        array_path = "0"
        additional_metadata = {}
    else:
        source_store = tmp_path / "input.zarr"
        source = f"{path_to_file_uri(source_store)}/image"
        output_store = tmp_path / "input_output.zarr"
        array_path = "image/0"
        additional_metadata = {
            "chunks": (1, 1, 1, 4, 4),
            "shards": (1, 1, 1, 8, 8),
        }

    regions = _image_regions(source, additional_metadata)
    writer = ZarrImageWriteStrategy(access=access_cls())
    writer.write_batch(tmp_path, regions[:2])
    assert not output_store.exists()
    writer.write_batch(tmp_path, regions[2:])

    output = zarr.open_array(output_store / array_path, mode="r")
    expected = np.stack(
        [np.full((2, 4, 8, 8), index, dtype=np.float32) for index in range(3)]
    )
    np.testing.assert_array_equal(output[:], expected)
    if source_kind == "zarr":
        assert output.chunks == (1, 1, 1, 4, 4)
        assert output.shards == (1, 1, 1, 8, 8)

    validate_zarr_store(output_store)


def test_write_t_only_whole_image(tmp_path: Path) -> None:
    """Test that all T samples are written."""
    regions = [
        ImageRegionData(
            data=np.full((2, 4, 8, 8), sample_idx, dtype=np.float32),
            source="array",
            data_shape=(3, 2, 4, 8, 8),
            dtype="float32",
            axes="TCZYX",
            target_axes="TCZYX",
            original_data_shape=(3, 2, 4, 8, 8),
            region_spec={
                "data_idx": 0,
                "sample_idx": sample_idx,
                "coords": (0, 0, 0),
                "patch_size": (8, 8, 8),
            },
            additional_metadata={},
        )
        for sample_idx in range(3)
    ]

    # check that it waits for all Ts to be cached
    output_store = tmp_path / "array_0.ome.zarr"
    writer = ZarrImageWriteStrategy()
    writer.write_batch(tmp_path, regions[:2])
    assert not output_store.exists()
    writer.write_batch(tmp_path, regions[2:])

    output = zarr.open_array(output_store / "0", mode="r")
    expected = np.stack(
        [np.full((2, 4, 8, 8), index, dtype=np.float32) for index in range(3)]
    )
    np.testing.assert_array_equal(output[:], expected)
    validate_zarr_store(output_store)


@pytest.mark.parametrize("source_kind", ["array", "file"])
def test_write_single_zarrs(tmp_path: Path, source_kind: str) -> None:
    """test that the writer writes separate whole-image Zarr stores."""
    if source_kind == "array":
        sources = ["array", "array"]
        output_stores = [
            tmp_path / "array_0.ome.zarr",
            tmp_path / "array_1.ome.zarr",
        ]
    else:
        sources = [str(tmp_path / f"input_{i}.tiff") for i in range(2)]
        output_stores = [
            tmp_path / "input_0.ome.zarr",
            tmp_path / "input_1.ome.zarr",
        ]

    regions = [
        region
        for data_idx, source in enumerate(sources)
        for region in _image_regions(source, {}, data_idx=data_idx)
    ]

    writer = ZarrImageWriteStrategy()
    writer.write_batch(tmp_path, regions)

    expected = np.stack(
        [np.full((2, 4, 8, 8), index, dtype=np.float32) for index in range(3)]
    )
    for output_store in output_stores:
        output = zarr.open_array(output_store / "0", mode="r")
        np.testing.assert_array_equal(output[:], expected)
        validate_zarr_store(output_store)


@pytest.mark.parametrize("source_kind", ["array", "file"])
def test_write_st_collection(tmp_path: Path, source_kind: str) -> None:
    """Test that ST predictions are split into one OME image per S plane."""
    source = "array" if source_kind == "array" else str(tmp_path / "input.tiff")
    regions = [
        ImageRegionData(
            data=np.full((1, 8, 8), sample_idx, dtype=np.float32),
            source=source,
            data_shape=(2, 3, 8, 8),
            dtype="float32",
            axes="STYX",
            target_axes="STYX",
            original_data_shape=(2, 3, 8, 8),
            region_spec={
                "data_idx": 0,
                "sample_idx": sample_idx,
                "coords": (0, 0),
                "patch_size": (8, 8),
            },
            additional_metadata={},
        )
        for sample_idx in range(6)
    ]

    output_store = (
        tmp_path / "array_0.ome.zarr"
        if source_kind == "array"
        else tmp_path / "input.ome.zarr"
    )
    ZarrImageWriteStrategy().write_batch(tmp_path, regions)

    root = zarr.open_group(output_store, mode="r")
    assert sorted(root["OME"].attrs["ome"]["series"]) == ["0", "1"]
    for sample_idx in range(2):
        output = root[f"{sample_idx}/0"]
        assert output.shape == (3, 8, 8)
        expected = np.arange(sample_idx * 3, sample_idx * 3 + 3)[:, None, None]
        np.testing.assert_array_equal(
            output[:], np.broadcast_to(expected, output.shape)
        )

    validate_zarr_store(output_store)
