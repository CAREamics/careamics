import numpy as np
import pytest

from careamics.config.tile_information import TileInformation


def test_defaults():
    """Test instantiating time information with defaults."""
    tile_info = TileInformation(
        array_shape=np.zeros((6, 6)).shape,
        overlap_crop_coords=((1, 2),),
        stitch_coords=((3, 4),),
    )

    assert tile_info.array_shape == (6, 6)
    assert not tile_info.last_tile


def test_error_on_coords():
    """Test than an error is raised if no coordinates are given."""
    with pytest.raises(ValueError):
        TileInformation(array_shape=(6, 6))


def test_error_on_singleton_dims():
    """Test that an error is raised if the array shape contains singleton dimensions."""
    with pytest.raises(ValueError):
        TileInformation(
            array_shape=(2, 1, 6, 6),
            overlap_crop_coords=((1, 2),),
            stitch_coords=((3, 4),),
        )


def test_tile_equality():
    """Test whether two tile information objects are equal."""
    t1 = TileInformation(
        array_shape=(6, 6),
        last_tile=True,
        overlap_crop_coords=((1, 2),),
        stitch_coords=((3, 4),),
    )
    t2 = TileInformation(
        array_shape=(6, 6),
        last_tile=True,
        overlap_crop_coords=((1, 2),),
        stitch_coords=((3, 4),),
    )
    assert t1 == t2

    # inequality
    t2.array_shape = (7, 7)
    assert t1 != t2

    t2.array_shape = (6, 6)
    t2.last_tile = False
    assert t1 != t2

    t2.last_tile = True
    t2.overlap_crop_coords = ((2, 3),)
    assert t1 != t2

    t2.overlap_crop_coords = ((1, 2),)
    t2.stitch_coords = ((4, 5),)
    assert t1 != t2
