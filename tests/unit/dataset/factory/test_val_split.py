import itertools
from collections.abc import Sequence
from contextlib import nullcontext

import numpy as np
import pytest

from careamics.dataset.factory.val_split import (
    _block_sequence,
    _contains_gap_size_1,
    _coord_is_removable,
    _create_validation_blocks,
    _find_block_sequence_params,
    _n_viable_val_patches,
    _remove_excess_selected,
    create_val_split,
    select_validation,
)
from careamics.dataset.patching import StratifiedPatching


def _gap_size_1(
    coords: np.ndarray,
    grid_shape: Sequence[int],
) -> bool:
    """Helper to check if there is a gap size of 1 between coordinates.

    This includes a gap size of 1 between any coordinate and the edge of the grid.

    Parameters
    ----------
    coords: numpy.ndarray
        Coordinates on a grid.
    grid_shape: Sequence[int]
        The shape of the grid.

    Returns
    -------
    bool
        Whether there is a gap size of one.
    """
    coord_map = np.zeros(grid_shape, dtype=bool)
    coord_map[tuple(coords[:, dim] for dim in range(len(grid_shape)))] = True
    coord_map = np.pad(coord_map, 2, mode="constant", constant_values=True)

    for axis in range(coord_map.ndim):
        lines = np.moveaxis(coord_map, axis, -1).reshape(-1, coord_map.shape[axis])
        for line in lines:
            if _contains_gap_size_1(line):
                return True
    return False


def _assert_no_gap_size_1(
    coords: np.ndarray,
    grid_shape: Sequence[int],
) -> None:
    """Helper to assert there is not a gap size of between the coordinates."""
    assert not _gap_size_1(coords, grid_shape)


def test_no_gap_size_1():
    """Test the helper `_gap_size_1` can detect a gap size of 1."""

    # --- test has gap of size 1
    # slices that will make a gap of size 1
    add_gap_slices = [
        (0, 0, 0),
        (1, 1, 1),
        (2, 2, 1),
        (0, 0, slice(None)),
        (1, slice(None), 1),
        (slice(None), 0, 2),
        (slice(None), slice(None), 1),
        (slice(None), 0, slice(None)),
        (2, slice(None), slice(None)),
    ]
    for s in add_gap_slices:
        grid_shape = (3, 3, 3)
        array = np.ones(grid_shape, dtype=bool)
        # add a gap size of 1
        array[*s] = False
        coords = np.stack(np.where(array), axis=-1)
        assert _gap_size_1(coords, grid_shape)

    # --- test does not have gap of size 1
    no_gap_slices = [
        (slice(0, 0), slice(0, 0), slice(0, 0)),
        (slice(None), slice(None), slice(None)),
        (0, 2, slice(None)),
        (0, slice(None), 0),
        (slice(None), 2, 2),
    ]
    for s in no_gap_slices:
        grid_shape = (3, 3, 3)
        array = np.zeros(grid_shape, dtype=bool)
        # create an array with no gaps of size 1
        array[*s] = True
        coords = np.stack(np.where(array), axis=-1)
        assert not _gap_size_1(coords, grid_shape)


@pytest.mark.parametrize(
    ("data_shapes", "n_val_patches", "expected_error"),
    list(
        itertools.product(
            [
                # data_shapes 2D
                [(1, 1, 512, 512)],
                [(4, 1, 256, 256)],
                [(1, 1, 128, 128), (4, 1, 256, 256)],
            ],
            [0, 1, 2, 8, 16],
            [nullcontext(0)],
        )
    )
    + list(
        itertools.product(
            [
                # data_shapes 3D
                [(1, 1, 256, 512, 512)],
                [(4, 1, 256, 256, 256)],
                [(1, 1, 128, 128, 128), (4, 1, 256, 256, 256)],
                [(1, 1, 64, 512, 512)],  # small Z should be allowed
            ],
            [0, 1, 2, 8, 16],
            [nullcontext(0)],
        )
    )
    # shapes too small
    + [
        ([(1, 1, 128, 128)], 1, pytest.raises(ValueError, match="validation patches")),
        (
            [(1, 1, 128, 128, 128)],
            1,
            pytest.raises(ValueError, match="validation patches"),
        ),
        ([(1, 1, 512, 512)], 64, pytest.raises(ValueError, match="validation patches")),
        (
            [(1, 1, 256, 512, 512)],
            256,
            pytest.raises(ValueError, match="validation patches"),
        ),
    ],
)
def test_create_val_split(
    data_shapes: Sequence[Sequence[int]],
    n_val_patches: int,
    expected_error,
):
    # NOTE: the functional tests ensure that validation does not overlap with training
    """
    Test the validation and training patching strategies have appropriate N patches.
    """
    ndims = len(data_shapes[0]) - 2
    patch_size = (64,) * ndims
    rng = np.random.default_rng(42)
    stratified_patching = StratifiedPatching(data_shapes, patch_size, seed=42)
    with expected_error:
        train_patching, val_patching = create_val_split(
            stratified_patching, n_val_patches, rng
        )
        assert train_patching.n_patches >= 1
        assert val_patching.n_patches >= n_val_patches


@pytest.mark.parametrize(
    ("grid_shape", "n_val_patches"),
    list(
        itertools.product([(8, 8), (7, 16), (7, 8, 8), (2, 8, 8)], [1, 8, 12, 15, 21])
    ),
)
def test_select_validation(grid_shape: tuple[int, int], n_val_patches: int):
    """Test `select_validation` selects an appropriate number of patches."""
    coords = select_validation(grid_shape, n_val_patches, np.random.default_rng(0))

    # desired val patches selected (won't always be exact)
    assert n_val_patches <= len(coords) < np.prod(grid_shape)
    # all coords are on the grid
    assert all(
        all([0 <= c and c <= gs] for c, gs in zip(coord, grid_shape, strict=True))
        for coord in coords
    )


@pytest.mark.parametrize(
    ("grid_shape", "n_val_patches"),
    list(
        itertools.product([(8, 8), (7, 16), (7, 8, 8), (2, 8, 8)], [1, 8, 12, 15, 21])
    ),
)
def test_create_validation_blocks(grid_shape: tuple[int, ...], n_val_patches: int):
    """Test the validation blocks do not have gap sizes of 1."""
    coords = _create_validation_blocks(
        grid_shape, n_val_patches, np.random.default_rng(0)
    )

    # test no gaps of size 1
    _assert_no_gap_size_1(coords, grid_shape)

    # test more than or equal to validation patches
    assert len(coords) >= n_val_patches
    assert len(coords) <= _n_viable_val_patches(grid_shape)


@pytest.mark.parametrize(
    ("block_size", "gap_size", "max_value"),
    list(itertools.product([1, 2, 3, 4, 5], [2, 3, 4, 5], [5, 10, 20])),
)
def test_block_sequence(block_size: int, gap_size: int, max_value: int):
    """Test `_block_sequence` produces periodic sequences of consecutive numbers."""
    sequence = _block_sequence(
        block_size=block_size, gap_size=gap_size, max_value=max_value
    )
    assert sequence[-1] < max_value
    # turn sequence into bools
    bool_sequence = np.zeros(max_value, dtype=bool)
    bool_sequence[sequence] = True

    period = block_size + gap_size
    for p in range(0, max_value, period):
        # block segment should be all true
        assert bool_sequence[p : p + block_size].all()
        # gap segment should be all false
        assert not bool_sequence[p + block_size : p + block_size + gap_size].any()


@pytest.mark.parametrize(
    ("max_value", "n_val_patches"), itertools.product([8, 13, 20], [0, 1, 2, 5, 6])
)
def test_find_block_sequence_params(max_value: int, n_val_patches: int):
    """Test that the required constraints of the validation sequence are not broken."""
    expected_error = (
        pytest.raises(ValueError, match="n_values=0")
        if n_val_patches == 0
        else nullcontext(0)
    )
    with expected_error:
        block_size, gap_size = _find_block_sequence_params(max_value, n_val_patches)
        sequence = _block_sequence(block_size, gap_size, max_value)

        # gap size always greater than 2
        assert gap_size >= 2

        # edge gap
        assert max_value - 1 - sequence[-1] != 1

        # greater than n_val_patches
        assert len(sequence) >= n_val_patches


@pytest.mark.parametrize(
    ("grid_shape", "n_val_patches"),
    list(
        itertools.product([(8, 8), (7, 16), (7, 8, 8), (2, 8, 8)], [1, 3, 13, 17, 21])
    ),
)
def test_remove_excess_selected(grid_shape: tuple[int, int], n_val_patches: int):
    """Test that removing excess patches does not result in a gap size of 1."""
    rng = np.random.default_rng(42)
    coords = _create_validation_blocks(grid_shape, n_val_patches, rng)
    selected = _remove_excess_selected(grid_shape, coords, n_val_patches, rng)

    # no gaps less than size 1
    _assert_no_gap_size_1(selected, grid_shape)

    # result is not less than n_val_patches
    # removability is not guaranteed ...
    assert len(selected) >= n_val_patches


def test_coord_is_removable():
    """Test that `_coord_is_removable` correctly identifies removable coords."""

    coord_map = np.array(
        [
            [True, True, True, False, False, True],
            [True, True, True, False, False, True],
            [True, True, True, False, False, True],
            [False, False, False, False, False, False],
        ]
    )
    coords = np.where(coord_map)
    removable_coords = [(2, 2), (2, 5)]
    coord_map = np.pad(coord_map, 2, mode="constant", constant_values=True)
    for coord in zip(*coords, strict=True):
        if tuple(int(c) for c in coord) in removable_coords:
            assert _coord_is_removable(np.array(coord), coord_map, padding=2)
        else:
            assert not _coord_is_removable(np.array(coord), coord_map, padding=2)


@pytest.mark.parametrize(
    ("arr", "expected"),
    [
        (np.array([True, False, True]), True),
        (np.array([True, True, False, True]), True),
        (np.array([True, False, False, True]), False),
        (np.array([True, True, False, False]), False),
    ],
)
def test_contains_gap_size_1(arr: np.ndarray, expected: bool):
    # examples
    assert _contains_gap_size_1(arr) is expected


@pytest.mark.parametrize(
    ("grid_shape", "expected"),
    [
        ((2, 2), 0),
        ((2, 2, 2), 0),
        ((8, 8), 36),
        ((1, 8), 6),
        ((2, 8), 12),
        ((8, 8, 8), 216),
        ((2, 8, 8), 72),
        ((1, 8, 8), 36),
    ],
)
def test_n_viable_val_patches(grid_shape: Sequence[int], expected: int):
    n_viable = _n_viable_val_patches(grid_shape)
    assert n_viable == expected
