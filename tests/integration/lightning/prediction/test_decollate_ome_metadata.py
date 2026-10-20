import numpy as np
from torch.utils.data import default_collate
from yaozarrs import v05

from careamics.dataset.image_region_data import ImageRegionData
from careamics.dataset.image_stack.zarr_access import build_ome_metadata
from careamics.lightning.prediction.decollate_utils import (
    decollate_image_region_data,
)


def _ome_additional_metadata(index: int) -> dict:
    ome_metadata = v05.Image(
        multiscales=[
            v05.Multiscale(
                name=f"img_{index}",
                axes=[
                    v05.ChannelAxis(name="c"),
                    v05.SpaceAxis(name="y", unit="micrometer"),
                    v05.SpaceAxis(name="x", unit="micrometer"),
                ],
                datasets=[
                    v05.Dataset(
                        path=str(index),
                        coordinateTransformations=[
                            v05.ScaleTransformation(
                                scale=[1.0, 0.5 + index, 0.25 + index]
                            )
                        ],
                    )
                ],
            )
        ]
    )
    return build_ome_metadata(
        image_group_path=f"img_{index}",
        level=str(index),
        ome_metadata=ome_metadata,
    ).to_additional_metadata()


def _region(source: str, index: int) -> ImageRegionData:
    return ImageRegionData(
        data=np.full((1, 8, 8), fill_value=index, dtype=np.float32),
        source=source,
        data_shape=(1, 8, 8),
        dtype="float32",
        axes="CYX",
        target_axes="CYX",
        original_data_shape=(1, 8, 8),
        region_spec={
            "data_idx": index,
            "sample_idx": index,
            "coords": (index, index + 1),
            "patch_size": (8, 8),
        },
        additional_metadata={
            "chunks": (1, 4, 4),
            "shards": (1, 8, 8),
            **_ome_additional_metadata(index),
        },
    )


def test_decollate_preserves_ome_additional_metadata() -> None:
    batch = default_collate(
        [
            _region("file:///tmp/sample_0.zarr/img_0/0", 0),
            _region("file:///tmp/sample_1.zarr/img_1/1", 1),
        ]
    )

    decollated = decollate_image_region_data(batch)

    assert len(decollated) == 2

    for idx in range(2):
        first = decollated[idx]
        assert first.source == f"file:///tmp/sample_{idx}.zarr/img_{idx}/{idx}"
        assert first.additional_metadata["chunks"] == (1, 4, 4)
        assert first.additional_metadata["shards"] == (1, 8, 8)
        assert first.additional_metadata["ome"]["layout"] == "collection"
        assert first.additional_metadata["ome"]["image_group_path"] == f"img_{idx}"
        assert first.additional_metadata["ome"]["level"] == str(idx)
        assert first.additional_metadata["ome"]["axes"] == [
            {"name": "c", "type": "channel"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ]
        assert tuple(
            first.additional_metadata["ome"]["coordinate_transformations"][0]["scale"]
        ) == (1.0, 0.5 + idx, 0.25 + idx)
