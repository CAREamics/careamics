"""Configuration for COSDD."""

from typing import Self

from pydantic import model_validator

from careamics.config.algorithms import COSDDAlgorithm
from careamics.config.augmentations import XYRandomRotate90Config
from careamics.config.data.data_config import Mode

from .configuration import Configuration


class COSDDConfiguration(Configuration):
    """COSDD-specific configuration."""

    algorithm_config: COSDDAlgorithm

    @model_validator(mode="after")
    def target_axes_must_be_none(self: Self) -> Self:
        """Ensure COSDD does not use separate target axes.

        Returns
        -------
        Self
            Validated configuration.

        Raises
        ------
        ValueError
            If `target_axes` is not `None`.
        """
        if self.data_config.target_axes is not None:
            raise ValueError("COSDD does not support non-None `target_axes`.")
        return self

    @model_validator(mode="after")
    def validate_patch_against_model(self: Self) -> Self:
        """Validate that the training patch size matches the model input shape.

        The size of the top prior of the Ladder VAE is derived from the training
        patch size. Prediction can be performed on any spatial size.

        Returns
        -------
        Self
            Validated configuration.

        Raises
        ------
        ValueError
            If the training patch size differs from the model input shape.
        """
        if self.data_config.mode != Mode.TRAINING:
            return self

        patch_size = tuple(self.data_config.patching.patch_size)  # type: ignore[union-attr]
        if patch_size != tuple(self.algorithm_config.model.input_shape):
            raise ValueError(
                f"The training patch size ({patch_size}) must be equal to the COSDD "
                f"model input shape ({self.algorithm_config.model.input_shape})."
            )
        return self

    @model_validator(mode="after")
    def validate_rotation_against_noise_direction(self: Self) -> Self:
        """Validate that 90 degree rotations do not change the noise direction.

        Returns
        -------
        Self
            Validated configuration.

        Raises
        ------
        ValueError
            If random 90 degree rotations are used with noise correlated along X or
            Y.
        """
        noise_direction = self.algorithm_config.model.noise_direction
        if noise_direction in ("x", "y") and any(
            isinstance(aug, XYRandomRotate90Config)
            for aug in self.data_config.augmentations
        ):
            raise ValueError(
                f"Random 90 degree rotations swap the X and Y axes and are not "
                f"compatible with noise correlated along {noise_direction}. Remove "
                f"the rotation augmentation."
            )
        return self

    @model_validator(mode="after")
    def warn_batch_norm(self: Self) -> Self:
        """Skip the batch normalization warning of the parent configuration.

        COSDD uses batch normalization with small batches and gradient accumulation
        by design.

        Returns
        -------
        Self
            Validated configuration.
        """
        return self

    def set_3D(self, is_3D: bool, axes: str, patch_size: list[int]) -> None:
        """
        Set 3D flag and axes.

        The COSDD model dimensionality is set by its input shape, which is updated to
        the new patch size.

        Parameters
        ----------
        is_3D : bool
            Whether the algorithm is 3D or not.
        axes : str
            Axes of the data.
        patch_size : list[int]
            Patch size.

        Raises
        ------
        ValueError
            If `is_3D` does not match the length of `patch_size`.
        """
        if is_3D != (len(patch_size) == 3):
            raise ValueError(
                f"`is_3D={is_3D}` does not match the patch size {patch_size}."
            )
        self.algorithm_config.model.input_shape = tuple(patch_size)
        self.data_config.set_3D(axes, patch_size)
        COSDDConfiguration.model_validate(self)
