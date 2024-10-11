from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from pylibCZIrw import czi as pyczi

from careamics.dataset.czi_dataset_utils import (
    extract_random_czi_patches,
    extract_sequential_czi_patches,
    get_czi_shape,
    iterate_file_names,
    plane_iterator,
)
from careamics.file_io.read import read_czi_roi


# Mocking the CziReader for unit tests
@pytest.fixture
def mock_czi_reader():
    mock_reader = MagicMock()
    mock_reader.total_bounding_box = {
        "Z": [0, 5],
        "T": [0, 10],
        "X": [0, 100],
        "Y": [0, 100],
    }
    return mock_reader


def test_iterate_file_names():
    data_files = [Path(f"file_{i}.czi") for i in range(4)]
    target_files = [Path(f"file_{i}.czi") for i in range(4)]

    # Test with matching files
    results = list(iterate_file_names(data_files, target_files))
    assert len(results) == len(data_files)
    for data, target in results:
        assert data.name == target.name

    # Test with non-matching files
    target_files[1] = Path("mismatched_file.czi")
    with pytest.raises(ValueError):
        list(iterate_file_names(data_files, target_files))


def test_get_czi_shape(mock_czi_reader):
    shape = get_czi_shape(mock_czi_reader)
    assert shape == [10, 100, 100]  # Based on the mocked bounding box and T>Z, TXY
    mock_czi_reader.total_bounding_box.pop("T", None)
    shape = get_czi_shape(mock_czi_reader)
    assert shape == [5, 100, 100]  # ZXY as T is removed
    # Test when Z and T are not present in total_bounding_box
    mock_czi_reader.total_bounding_box.pop("Z", None)
    shape = get_czi_shape(mock_czi_reader)
    assert shape == [100, 100]  # XY as T and Z not there in the reader


def test_plane_iterator(mock_czi_reader):
    # Test with 2D patch
    patch_size = (256, 256)
    # With C dimension present
    mock_czi_reader.total_bounding_box["C"] = (0, 3)
    _, key_list, updated_patch_size = plane_iterator(mock_czi_reader, patch_size)
    assert len(key_list) == 4  # Expecting Z,T, X and Y dimensions
    assert updated_patch_size == (3, 1, 256, 256)  # patch size CZXY
    assert "C" not in key_list  # C will be poped out if its present in the reader

    # Without C dimension
    _, key_list, updated_patch_size = plane_iterator(mock_czi_reader, patch_size)
    assert len(key_list) == 4  # Expecting Z,T, X and Y dimensions
    assert updated_patch_size == (1, 1, 256, 256)

    # Test with 3D patch_size
    patch_size = (5, 256, 256)
    # With C dimension present
    mock_czi_reader.total_bounding_box["C"] = (0, 3)
    _, key_list, updated_patch_size = plane_iterator(mock_czi_reader, patch_size)
    assert len(key_list) == 3  # Expecting  Z/T and XY dimensions
    assert updated_patch_size == (3, 5, 256, 256)  # patch size CZXY
    assert "C" not in key_list
    assert "T" not in key_list  # as T is greater than Z in the mocked example


def test_extract_sequential_czi_patches(tmp_path):
    # create a sample czi file (CZTXY, C=3,Z=10,T=20)
    file = tmp_path / "sample.czi"
    planes = [
        {"C": c, "T": t, "Z": z} for c in range(3) for z in range(10) for t in range(20)
    ]
    with pyczi.create_czi(
        file, exist_ok=True, compression_options="uncompressed:"
    ) as czidoc_w:
        for plane in planes:
            czidoc_w.write(
                data=np.random.rand(10, 10).astype(np.float32),
                plane=plane,
            )

    patches = list(
        extract_sequential_czi_patches(
            read_source_func=read_czi_roi, czi_file_path=Path(file)
        )
    )

    assert len(patches) > 0

    for patch in patches:
        assert patch.shape[0] == 3
        assert isinstance(patch, np.ndarray)


def test_extract_random_czi_patches(tmp_path):
    # create a sample czi file (CXY, C=3)
    file = tmp_path / "sample.czi"
    planes = [
        {
            "C": c,
            "T": 0,
        }
        for c in range(3)
    ]
    with pyczi.create_czi(
        file, exist_ok=True, compression_options="uncompressed:"
    ) as czidoc_w:
        for plane in planes:
            czidoc_w.write(
                data=np.random.rand(10, 10).astype(np.float32),
                plane=plane,
            )

    # Extract 2d patch (each patch should be (C,*patch_size))
    patch_size = (10, 10)
    patches = list(
        extract_random_czi_patches(file, patch_size, read_source_func=read_czi_roi)
    )
    assert len(patches) > 0
    for patch in patches:
        assert (
            patch[0].shape[0] == 3
        )  # 2D patch and data with C dimension return CXY patch
        assert isinstance(patch[0], np.ndarray)
        assert patch[1] is None  # since target is None

    # Extract 3D patches from the sample data which does not have Z/T axis. Raises error
    patch_size = (5, 10, 10)
    with pytest.raises(ValueError):
        patches = list(
            extract_random_czi_patches(file, patch_size, read_source_func=read_czi_roi)
        )

    # create a sample with Z,T and C axis
    planes = [
        {"C": c, "T": t, "Z": z} for c in range(3) for t in range(20) for z in range(10)
    ]
    with pyczi.create_czi(
        file, exist_ok=True, compression_options="uncompressed:"
    ) as czidoc_w:
        for plane in planes:
            czidoc_w.write(data=np.random.rand(10, 10).astype(np.float32), plane=plane)

    patch_size = (5, 10, 10)
    patches = list(
        extract_random_czi_patches(file, patch_size, read_source_func=read_czi_roi)
    )
    assert len(patches) > 0
    for patch in patches:
        assert (
            patch[0].shape[0] == 3
        )  # 3D patch and data with C dimension return CZXY patch
        assert (
            patch[0].shape[1] == 5
        )  # 3D patch and data with C dimension return CZXY patch
        assert isinstance(patch[0], np.ndarray)
        assert patch[1] is None  # since target is None

    # try to extract larger deth patch than the Z or T dimension
    patch_size = (25, 10, 10)
    with pytest.raises(ValueError):
        patches = list(
            extract_random_czi_patches(file, patch_size, read_source_func=read_czi_roi)
        )
