import numpy as np
import pytest

from careamics.config.tile_information import TileInformation


def test_defaults():
    """Test instantiating time information with defaults."""
    tile_info = TileInformation(array_shape=np.zeros((6, 6)).shape)

    assert tile_info.array_shape == (6, 6)
    assert not tile_info.tiled
    assert not tile_info.last_tile
    assert tile_info.overlap_crop_coords is None
    assert tile_info.stitch_coords is None


def test_tiled():
    """Test instantiating time information with parameters."""
    tile_info = TileInformation(
        array_shape=np.zeros((6, 6)).shape,
        tiled=True,
        last_tile=True,
        overlap_crop_coords=((1, 2),),
        stitch_coords=((3, 4),),
    )

    assert tile_info.array_shape == (6, 6)
    assert tile_info.tiled
    assert tile_info.last_tile
    assert tile_info.overlap_crop_coords == ((1, 2),)
    assert tile_info.stitch_coords == ((3, 4),)


def test_validation_last_tile():
    """Test that last tile is only set if tiled is set."""
    tile_info = TileInformation(array_shape=(6, 6), last_tile=True)
    assert not tile_info.last_tile


def test_error_on_coords():
    """Test than an error is raised if it is tiled but not coordinates are given."""
    with pytest.raises(ValueError):
        TileInformation(array_shape=(6, 6), tiled=True)


def test_error_on_singleton_dims():
    """Test that an error is raised if the array shape contains singleton dimensions."""
    with pytest.raises(ValueError):
        TileInformation(array_shape=(2, 1, 6, 6))
