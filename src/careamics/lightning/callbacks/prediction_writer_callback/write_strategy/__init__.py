"""Write strategies for the prediciton writer callback."""

__all__ = [
    "WriteStrategy",
    "CacheTiles",
    "WriteImage",
    "WriteTilesZarr",
]

from .cache_tiles import CacheTiles
from .protocol import WriteStrategy
from .write_image import WriteImage
from .write_tiles_zarr import WriteTilesZarr
