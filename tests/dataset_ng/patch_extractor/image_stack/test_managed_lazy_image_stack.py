from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import tifffile

from careamics.dataset.dataset_utils import reshape_array
from careamics.dataset_ng.patch_extractor.image_stack import ManagedLazyImageStack


@pytest.mark.parametrize(
    "original_axes, original_shape, expected_shape, sample_idx",
    [
        ("YX", (32, 48), (1, 1, 32, 48), 0),
        ("XYS", (48, 32, 3), (3, 1, 32, 48), 1),
        ("SXYC", (3, 48, 32, 2), (3, 2, 32, 48), 1),
        ("CYXT", (2, 32, 48, 3), (3, 2, 32, 48), 2),
        ("CXYTS", (2, 48, 32, 3, 2), (6, 2, 32, 48), 4),
        ("XCSYT", (48, 1, 2, 32, 3), (6, 1, 32, 48), 5),  # crazy one
    ],
)
def test_extract_patch_2D(
    tmp_path: Path,
    original_axes: str,
    original_shape: tuple[int, ...],
    expected_shape: tuple[int, ...],
    sample_idx: int,
):
    data = np.arange(np.prod(original_shape)).reshape(original_shape)
    data_ref = reshape_array(data, original_axes)
    path = Path(tmp_path / "image.tiff")
    tifffile.imwrite(path, data)

    # test extracted patch matches patch from reference data
    coords = (11, 4)
    patch_size = (16, 9)

    on_load = Mock()
    on_close = Mock()
    image_stack = ManagedLazyImageStack.from_tiff(
        path, original_axes, on_load=on_load, on_close=on_close
    )

    assert not image_stack.is_loaded

    extracted_patch = image_stack.extract_patch(
        sample_idx=sample_idx, coords=coords, patch_size=patch_size
    )
    patch_ref = data_ref[
        sample_idx,
        :,
        coords[0] : coords[0] + patch_size[0],
        coords[1] : coords[1] + patch_size[1],
    ]
    np.testing.assert_array_equal(extracted_patch, patch_ref)

    assert image_stack.is_loaded
    on_load.assert_called_once()

    image_stack.deallocate()
    assert not image_stack.is_loaded
    on_close.assert_called_once()
