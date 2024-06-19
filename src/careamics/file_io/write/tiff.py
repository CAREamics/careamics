from pathlib import Path

import tifffile
from numpy.typing import NDArray


def write_tiff(file_path: Path, img: NDArray, *args, **kwargs) -> None:
    tifffile.imwrite(file_path, img, *args, **kwargs)
