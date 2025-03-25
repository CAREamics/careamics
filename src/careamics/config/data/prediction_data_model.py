"""Prediction data configuration."""

from typing import Any, Optional, Union

from pydantic import Field

from .general_data_model import DataConfig


class PredictionDataConfig(DataConfig):
    """Prediction data configuration.

    The following parameters defined in the parent `DataConfig` are aliased:
    - `path_size` -> `tile_size`
    - `patch_overlap` -> `tile_overlap`
    - `train_dataloader_params` -> `dataloader_params`

    The aliases should be used to pass the parameters to the model.

    As opposed to `DataConfig`, the `val_dataloader_params` is set to `None` by default.
    Likewise, `patche_size` (`tile_size`) and `patch_overlap` (`tile_overlap`) can be
    `None` to predict on the entire image at once.

    If `transforms` are provided, then they are used as test-time augmentations (TTA).

    Examples
    --------
    Minimum example:

    >>> data = PredictionDataConfig(
    ...     data_type="array", # defined in SupportedData
    ...     batch_size=4,
    ...     axes="YX"
    ... )

    """

    patch_size: Optional[Union[list[int]]] = Field(
        default=None, min_length=2, max_length=3, alias="tile_size"
    )
    """Tile size used during prediction. If `None`, the entire image is predicted at
    once."""

    patch_overlaps: Optional[Union[list[int]]] = Field(
        default=[48, 48], min_length=2, max_length=3, alias="tile_overlaps"
    )
    """Overlap between tiles during prediction."""

    train_dataloader_params: dict[str, Any] = Field(
        default={}, validate_default=True, alias="dataloader_params"
    )
    """Dictionary of PyTorch prediction dataloader parameters."""

    val_dataloader_params: Optional[dict[str, Any]] = Field(default=None)
    """This parameter is unused during prediction."""

    # TODO: validate length patch size and patch overlap vs axes
