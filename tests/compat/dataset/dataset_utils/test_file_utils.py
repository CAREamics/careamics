from pathlib import Path

import numpy as np
import tifffile

from careamics.compat.dataset.dataset_utils.file_utils import (
    create_write_file_path,
    get_files_size,
)


def test_create_write_file_path():
    dirpath = Path("output_directory")
    file_path = Path("input_directory/file_name.in_ext")
    write_extension = ".out_ext"

    write_file_path = create_write_file_path(
        dirpath=dirpath, file_path=file_path, write_extension=write_extension
    )
    assert write_file_path == Path("output_directory/file_name.out_ext")


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
