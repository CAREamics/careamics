from pathlib import Path

from careamics.lightning.callbacks.prediction import (
    create_write_file_path,
)
from careamics.lightning.callbacks.prediction.file_path_utils import (
    common_source_base,
)


def test_create_write_file_path():
    dirpath = Path("output_directory")
    file_path = Path("input_directory/file_name.tif")
    write_extension = ".out_ext"

    write_file_path = create_write_file_path(
        dirpath=dirpath, file_path=file_path, write_extension=write_extension
    )
    assert write_file_path == Path("output_directory/file_name.out_ext")


def test_source_structure_prevents_collision():
    """Identically named files in different directories map to distinct outputs."""
    sources = [Path("dir1/image.tif"), Path("dir2/image.tif")]
    base = common_source_base(sources)

    outputs = [
        create_write_file_path(Path("pred"), source, ".tiff", source_base=base)
        for source in sources
    ]

    assert outputs == [Path("pred/dir1/image.tiff"), Path("pred/dir2/image.tiff")]
