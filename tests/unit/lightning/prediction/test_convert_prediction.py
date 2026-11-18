import numpy as np
from torch.utils.data import default_collate

from careamics.dataset.image_region_data import ImageRegionData
from careamics.lightning.prediction.convert_prediction import (
    decollate_image_region_data,
)


def _region(source: str, level: str) -> ImageRegionData:
    return ImageRegionData(
        data=np.zeros((1, 8, 8), dtype=np.float32),
        source=source,
        data_shape=(1, 8, 8),
        dtype="float32",
        axes="CYX",
        target_axes="CYX",
        original_data_shape=(1, 8, 8),
        region_spec={
            "data_idx": 0,
            "sample_idx": 0,
            "coords": (0, 0),
            "patch_size": (8, 8),
        },
        additional_metadata={
            "chunks": (1, 4, 4),
            "ome": {
                "layout": "collection",
                "image_group_path": f"sample_{level}",
                "level": level,
                "axes": [
                    {"name": "c", "type": "channel"},
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"},
                ],
                "coordinate_transformations": [
                    {"type": "scale", "scale": [1.0, 0.5, 0.5]}
                ],
                "version": "0.5",
            },
        },
    )


def test_decollate_image_region_data_preserves_nested_additional_metadata() -> None:
    batch = default_collate(
        [
            _region("file:///tmp/sample_0.zarr/0", "0"),
            _region("file:///tmp/sample_1.zarr/1", "1"),
        ]
    )

    decollated = decollate_image_region_data(batch)

    assert len(decollated) == 2
    assert decollated[0].additional_metadata["chunks"] == (1, 4, 4)
    assert decollated[0].additional_metadata["ome"]["layout"] == "collection"
    assert decollated[0].additional_metadata["ome"]["image_group_path"] == "sample_0"
    assert decollated[0].additional_metadata["ome"]["level"] == "0"
    assert decollated[0].additional_metadata["ome"]["axes"][1]["unit"] == "micrometer"
    assert tuple(
        decollated[0].additional_metadata["ome"]["coordinate_transformations"][0][
            "scale"
        ]
    ) == (1.0, 0.5, 0.5)
    assert decollated[1].additional_metadata["ome"]["level"] == "1"
