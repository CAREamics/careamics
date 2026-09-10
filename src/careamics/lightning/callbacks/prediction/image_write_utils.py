"""Shared utilities for writing complete images."""

from careamics.dataset.image_region_data import ImageRegionData


def _get_total_samples(prediction: ImageRegionData) -> int:
    """Get the expected total number of samples from data_shape and axes.

    Parameters
    ----------
    prediction : ImageRegionData
        A prediction containing metadata about the original data.

    Returns
    -------
    int
        Total number of samples in the S dimension, or 1 if no S dimension.
    """
    if "S" in prediction.axes:
        s_idx = prediction.axes.index("S")
        return prediction.data_shape[s_idx]
    return 1


def get_complete_images(image_cache: dict[int, list[ImageRegionData]]) -> list[int]:
    """
    Get data indices where all samples have been collected.

    Parameters
    ----------
    image_cache : dict[int, list[ImageRegionData]]
        Image cache, where list of ImageRegionData are indexed by their source data.

    Returns
    -------
    list of int
        Data indices of complete images in the cache.
    """
    complete_images = []
    for data_idx in image_cache.keys():
        total_samples = _get_total_samples(image_cache[data_idx][0])

        if len(image_cache[data_idx]) == total_samples:
            complete_images.append(data_idx)
        elif len(image_cache[data_idx]) > total_samples:
            raise ValueError(
                f"More samples cached for data_idx {data_idx} than expected. "
                f"Expected {total_samples}, found "
                f"{len(image_cache[data_idx])}."
            )

    return complete_images
