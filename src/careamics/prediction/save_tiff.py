from pathlib import Path
import tifffile

from numpy.typing import NDArray

def save_tiff(fp: Path, img: NDArray, *args, **kwargs) -> None:
    tifffile.imwrite(fp, img, *args, **kwargs)