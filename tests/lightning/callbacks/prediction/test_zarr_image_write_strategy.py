"""Tests for whole-image Zarr prediction writing."""

from pathlib import Path

import numpy as np
import pytest
import zarr
from yaozarrs import validate_zarr_store

from careamics.dataset.image_region_data import ImageRegionData
from careamics.dataset.image_stack.zarr_access import (
    ZarrPythonAccess,
    path_to_file_uri,
)
from careamics.lightning.callbacks.prediction import ZarrImageWriteStrategy

BACKENDS = [
    pytest.param(ZarrPythonAccess, id="zarr"),
]


def _image_regions(
    source: str,
    additional_metadata: dict,
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
                "data_idx": 0,
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
        output_store = tmp_path / "prediction.zarr"
        array_path = "0/0"
        additional_metadata = {}
    elif source_kind == "file":
        source = str(tmp_path / "input.tiff")
        output_store = tmp_path / "prediction.zarr"
        array_path = "input/0"
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
