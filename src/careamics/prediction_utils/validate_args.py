"""Module for validating prediction arguments."""


def validate_unet_tile_size(model_depth: int, tile_size: tuple[int, ...]):
    """
    Validate that `tile_size` is equal to `k*2^depth` where `k` is an integer.

    Parameters
    ----------
    model_depth : int
        The depth of the UNet model.
    tile_size : tuple of int
        The tile size.

    Raises
    ------
    ValueError
        If `tile_size` is not equal to `k*2^depth` where `k` is an integer.
    """
    # tile size must be equal to k*2^n,
    # where n is the number of pooling layers (equal to the depth) and k is an integer
    tile_increment = 2**model_depth
    for i, t in enumerate(tile_size):
        if t % tile_increment != 0:
            raise ValueError(
                f"Tile size must be divisible by {tile_increment} along "
                f"all axes (got {t} for axis {i}). If your image size is "
                f"smaller along one axis (e.g. Z), consider padding the "
                f"image."
            )
