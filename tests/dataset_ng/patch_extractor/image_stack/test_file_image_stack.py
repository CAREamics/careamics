from pathlib import Path

import numpy as np
import pytest
import tifffile

from careamics.dataset_ng.patch_extractor.image_stack import FileImageStack


def test_extract_patch(tmp_path: Path):
    data_shape = (64, 47)
    axes = "YX"
    data = np.arange(np.prod(data_shape)).reshape(data_shape)
    path = tmp_path / "image.tiff"
    tifffile.imwrite(path, data, metadata={"axes": axes})

    image_stack = FileImageStack.from_tiff(path, axes)

    # extract patch should raise an error if the image stack is not loaded
    with pytest.raises(ValueError):
        image_stack.extract_patch(sample_idx=0, coords=(4, 8), patch_size=(16, 16))

    # call load & extract patch
    image_stack.load()
    patch = image_stack.extract_patch(sample_idx=0, coords=(4, 8), patch_size=(16, 16))
    # confirm as expected against reference data
    np.testing.assert_array_equal(patch, data[np.newaxis, 4:20, 8:24])


def test_load_and_close(tmp_path: Path):
    data_shape = (64, 47)
    axes = "YX"
    data = np.arange(np.prod(data_shape)).reshape(data_shape)
    path = tmp_path / "image.tiff"
    tifffile.imwrite(path, data, metadata={"axes": axes})

    image_stack = FileImageStack.from_tiff(path, axes)
    assert not image_stack.is_loaded

    image_stack.load()
    assert image_stack.is_loaded

    image_stack.close()
    assert not image_stack.is_loaded
