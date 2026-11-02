from pathlib import Path

import mrcfile
import numpy as np

from careamics.dataset.image_stack import MrcImageStack
from careamics.dataset.patch_constructor.metadata_utils import get_image_metadata


def _write_mrc(path: Path, data: np.ndarray) -> None:
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(data)


def test_mrc_image_stack_extracts_memory_mapped_patch(tmp_path: Path):
    data = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    path = tmp_path / "volume.mrc"
    _write_mrc(path, data)

    image_stack = MrcImageStack(path, axes="ZYX")

    assert image_stack.source == path
    assert image_stack.original_axes == "ZYX"
    assert image_stack.original_data_shape == data.shape
    assert image_stack.data_shape == (1, 1, 4, 5, 6)
    assert image_stack.data_dtype == data.dtype
    assert not image_stack.is_loaded

    mrc_metadata = image_stack.additional_metadata["mrc"]
    assert mrc_metadata["nx"] == data.shape[2]
    assert mrc_metadata["ny"] == data.shape[1]
    assert mrc_metadata["nz"] == data.shape[0]
    assert "nsymbt" not in mrc_metadata
    assert "exttyp" not in mrc_metadata

    image_metadata = get_image_metadata(image_stack)
    assert image_metadata["additional_metadata"] == {"mrc": mrc_metadata}

    patch = image_stack.extract_patch(
        sample_idx=0,
        channels=None,
        coords=(1, 2, 3),
        patch_size=(2, 2, 2),
    )

    np.testing.assert_array_equal(patch, data[1:3, 2:4, 3:5][np.newaxis])
    assert image_stack.is_loaded

    image_stack.close()
    assert not image_stack.is_loaded
