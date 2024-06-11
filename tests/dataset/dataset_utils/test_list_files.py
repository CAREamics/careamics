from pathlib import Path

import numpy as np
import pytest
import tifffile

from careamics.config.support import SupportedData
from careamics.dataset.dataset_utils import (
    get_files_size,
    list_files,
    validate_source_target_files,
)


def test_get_files_size_tiff(tmp_path: Path):
    """Test getting size of multiple TIFF files."""
    # create array
    image = np.ones((10, 10))

    # save array to tiff
    path1 = tmp_path / "test1.tif"
    tifffile.imwrite(path1, image)

    path2 = tmp_path / "test2.tiff"
    tifffile.imwrite(path2, image)

    # save text file
    path3 = tmp_path / "test3.txt"
    path3.write_text("test")

    # save file in subdirectory
    subdirectory = tmp_path / "subdir"
    subdirectory.mkdir()
    path4 = subdirectory / "test3.tif"
    tifffile.imwrite(path4, image)

    # create file list
    files = [path1, path2, path4]

    # get files size
    size = get_files_size(files)
    assert size > 0


def test_list_single_file_tiff(tmp_path: Path):
    """Test listing a single TIFF file."""
    # create array
    image = np.ones((10, 10))

    # save array to tiff
    path = tmp_path / "test.tif"
    tifffile.imwrite(path, image)

    # list file using parent directory
    files = list_files(tmp_path, SupportedData.TIFF)
    assert len(files) == 1
    assert files[0] == path

    # list file using file path
    files = list_files(path, SupportedData.TIFF)
    assert len(files) == 1
    assert files[0] == path


def test_list_multiple_files_tiff(tmp_path: Path):
    """Test listing multiple TIFF files in subdirectories with additional files."""
    # create array
    image = np.ones((10, 10))

    # save array to /npy
    path1 = tmp_path / "test1.tif"
    tifffile.imwrite(path1, image)

    path2 = tmp_path / "test2.tif"
    tifffile.imwrite(path2, image)

    # save text file
    path3 = tmp_path / "test3.txt"
    path3.write_text("test")

    # save file in subdirectory
    subdirectory = tmp_path / "subdir"
    subdirectory.mkdir()
    path4 = subdirectory / "test3.tif"
    tifffile.imwrite(path4, image)

    # create file list
    ref_files = [path1, path2, path4]

    # list files using parent directory
    files = list_files(tmp_path, SupportedData.TIFF)
    assert len(files) == 3
    assert set(files) == set(ref_files)

    # test that the files are sorted
    assert files != ref_files
    assert files == sorted(ref_files)


def test_list_single_file_custom(tmp_path):
    """Test listing a single custom file."""
    # create array
    image = np.ones((10, 10))

    # save as .npy
    path = tmp_path / "custom.npy"
    np.save(path, image)

    # list files using parent directory
    files = list_files(tmp_path, SupportedData.CUSTOM)
    assert len(files) == 1
    assert files[0] == path

    # list files using file path
    files = list_files(path, SupportedData.CUSTOM)
    assert len(files) == 1
    assert files[0] == path


def test_list_multiple_files_custom(tmp_path: Path):
    """Test listing multiple custom files in subdirectories with additional files."""
    # create array
    image = np.ones((10, 10))

    # save array to /npy
    path1 = tmp_path / "test1.npy"
    np.save(path1, image)

    path2 = tmp_path / "test2.npy"
    np.save(path2, image)

    # save text file
    path3 = tmp_path / "test3.txt"
    path3.write_text("test")

    # save file in subdirectory
    subdirectory = tmp_path / "subdir"
    subdirectory.mkdir()
    path4 = subdirectory / "test3.npy"
    np.save(path4, image)

    # create file list (even the text file is selected)
    ref_files = [path1, path2, path3, path4]

    # list files using parent directory
    files = list_files(tmp_path, SupportedData.CUSTOM)
    assert len(files) == 4
    assert set(files) == set(ref_files)

    # list files using the file extension filter
    files = list_files(tmp_path, SupportedData.CUSTOM, "*.npy")
    assert len(files) == 3
    assert set(files) == {path1, path2, path4}


def test_validate_source_target_files(tmp_path: Path):
    """Test that it passes for two folders with same number of files and same names."""
    # create two subfolders
    src = tmp_path / "src"
    src.mkdir()

    tar = tmp_path / "tar"
    tar.mkdir()

    # populate with files
    filename_1 = "test1.txt"
    filename_2 = "test2.txt"

    (tmp_path / "src" / filename_1).write_text("test")
    (tmp_path / "tar" / filename_1).write_text("test")

    (tmp_path / "src" / filename_2).write_text("test")
    (tmp_path / "tar" / filename_2).write_text("test")

    # list files
    src_files = list_files(src, SupportedData.CUSTOM)
    tar_files = list_files(tar, SupportedData.CUSTOM)

    # validate files
    validate_source_target_files(src_files, tar_files)


def test_validate_source_target_files_wrong_names(tmp_path: Path):
    """Test that an error is raised if filenames are different."""
    # create two subfolders
    src = tmp_path / "src"
    src.mkdir()

    tar = tmp_path / "tar"
    tar.mkdir()

    # populate with files
    filename_1 = "test1.txt"
    filename_2 = "test2.txt"
    filename_3 = "test3.txt"

    (tmp_path / "src" / filename_1).write_text("test")
    (tmp_path / "tar" / filename_1).write_text("test")

    (tmp_path / "src" / filename_2).write_text("test")
    (tmp_path / "tar" / filename_3).write_text("test")

    # list files
    src_files = list_files(src, SupportedData.CUSTOM)
    tar_files = list_files(tar, SupportedData.CUSTOM)

    # validate files
    with pytest.raises(ValueError):
        validate_source_target_files(src_files, tar_files)


def test_validate_source_target_files_wrong_number(tmp_path: Path):
    """Test that an error is raised if filenames are different."""
    # create two subfolders
    src = tmp_path / "src"
    src.mkdir()

    tar = tmp_path / "tar"
    tar.mkdir()

    # populate with files
    filename_1 = "test1.txt"
    filename_2 = "test2.txt"

    (tmp_path / "src" / filename_1).write_text("test")
    (tmp_path / "tar" / filename_1).write_text("test")

    (tmp_path / "src" / filename_2).write_text("test")

    # list files
    src_files = list_files(src, SupportedData.CUSTOM)
    tar_files = list_files(tar, SupportedData.CUSTOM)

    # validate files
    with pytest.raises(ValueError):
        validate_source_target_files(src_files, tar_files)
