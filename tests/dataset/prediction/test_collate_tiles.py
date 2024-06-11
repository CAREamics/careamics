import pytest

from careamics.dataset.tiling import collate_tiles, extract_tiles


@pytest.mark.parametrize("n_channels", [1, 4])
@pytest.mark.parametrize("batch", [1, 3])
def test_collate_tiles_2d(ordered_array, n_channels, batch):
    """Test that the collate tiles function collates tile information correctly."""
    tile_size = (4, 4)
    tile_overlap = (2, 2)
    shape = (1, n_channels, 8, 8)

    # create array
    array = ordered_array(shape)

    # extract tiles
    tiles = list(extract_tiles(array, tile_size=tile_size, overlaps=tile_overlap))

    tiles_used = 0
    n_tiles = len(tiles)
    while tiles_used < n_tiles:
        # get a batch of tiles
        batch_tiles = tiles[tiles_used : tiles_used + batch]
        tiles_used += batch

        # collate the tiles
        collated_tiles = collate_tiles(batch_tiles)

        # check the collated tiles
        assert collated_tiles[0].shape == (batch, n_channels) + tile_size

        # check the tile info
        tile_infos = collated_tiles[1]
        assert len(tile_infos) == batch

        for i, t in enumerate(tile_infos):
            for j in range(i + 1, len(tile_infos)):
                assert t != tile_infos[j]


@pytest.mark.parametrize("n_channels", [1, 4])
@pytest.mark.parametrize("batch", [1, 3])
def test_collate_tiles_3d(ordered_array, n_channels, batch):
    """Test that the collate tiles function collates tile information correctly."""
    tile_size = (4, 4, 4)
    tile_overlap = (2, 2, 2)
    shape = (1, n_channels, 8, 8, 8)

    # create array
    array = ordered_array(shape)

    # extract tiles
    tiles = list(extract_tiles(array, tile_size=tile_size, overlaps=tile_overlap))

    tiles_used = 0
    n_tiles = len(tiles)
    while tiles_used < n_tiles:
        # get a batch of tiles
        batch_tiles = tiles[tiles_used : tiles_used + batch]
        tiles_used += batch

        # collate the tiles
        collated_tiles = collate_tiles(batch_tiles)

        # check the collated tiles
        assert collated_tiles[0].shape == (batch, n_channels) + tile_size

        # check the tile info
        tile_infos = collated_tiles[1]
        assert len(tile_infos) == batch

        for i, t in enumerate(tile_infos):
            for j in range(i + 1, len(tile_infos)):
                assert t != tile_infos[j]
