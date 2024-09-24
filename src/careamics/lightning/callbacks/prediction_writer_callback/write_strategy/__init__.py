"""Write strategies for the prediciton writer callback."""

__all__ = [
    "WriteStrategy",
    "WriteTiles",
    "WriteImage",
    "WriteTilesZarr",
]

from .protocol import WriteStrategy
from .write_image import WriteImage
from .write_tiles import WriteTiles
from .write_tiles_zarr import WriteTilesZarr
