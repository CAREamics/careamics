from pathlib import Path

import numpy as np
import zarr

from careamics.dataset.image_stack import InMemoryImageStack, ZarrImageStack
from careamics.dataset.image_stack.zarr_access import ZarrNode
from careamics.dataset.patch_constructor.metadata_utils import get_image_metadata


def test_in_memory_image_stack_additional_metadata_is_empty() -> None:
    image_stack = InMemoryImageStack.from_array(np.zeros((1, 8, 8)), axes="SYX")

    assert image_stack.additional_metadata == {}
    assert get_image_metadata(image_stack)["additional_metadata"] == {}


def test_zarr_image_stack_additional_metadata_contains_chunks_and_shards(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "data.zarr"
    group = zarr.create_group(store_path)
    group.create_array(
        "array_0",
        data=np.zeros((2, 8, 8), dtype=np.float32),
        chunks=(1, 4, 4),
        shards=(2, 8, 8),
    )

    image_stack = ZarrImageStack(
        node=ZarrNode(store_uri=store_path.as_uri(), path="array_0", node_type="array"),
        axes="SYX",
    )

    assert image_stack.additional_metadata == {
        "chunks": (1, 4, 4),
        "shards": (2, 8, 8),
    }
    assert get_image_metadata(image_stack)["additional_metadata"] == {
        "chunks": (1, 4, 4),
        "shards": (2, 8, 8),
    }
