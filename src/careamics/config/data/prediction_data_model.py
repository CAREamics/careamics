"""Prediction data configuration."""

from __future__ import annotations

from typing import Any, Optional, Union

from pydantic import Field, model_validator
from typing_extensions import Self

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

    # TODO: should we enforce `suffle=False` for prediction? does it matter?
    train_dataloader_params: dict[str, Any] = Field(
        default={}, validate_default=True, alias="dataloader_params"
    )
    """Dictionary of PyTorch prediction dataloader parameters."""

    val_dataloader_params: Optional[dict[str, Any]] = Field(default=None)
    """This parameter is unused during prediction."""

    @model_validator(mode="after")
    def validate_dimensions(self: Self) -> Self:
        """
        Validate 2D/3D dimensions between axes, patch size and patch overlaps.

        Returns
        -------
        Self
            Validated data model.

        Raises
        ------
        ValueError
            If the patch size and overlaps do not match axes.
        """
        if self.is_tiled():
            assert self.patch_size is not None
            assert self.patch_overlaps is not None

            if "Z" in self.axes:
                if len(self.patch_size) != 3 or len(self.patch_overlaps) != 3:
                    raise ValueError(
                        f"Patch size and overlaps must have 3 dimensions if the data is"
                        f" 3D (got axes {self.axes})."
                    )

            else:
                if len(self.patch_size) != 2 or len(self.patch_overlaps) != 2:
                    raise ValueError(
                        f"Patch size and overlaps must have 2 dimensions if the data is"
                        f" 2D (got axes {self.axes})."
                    )

        return self
