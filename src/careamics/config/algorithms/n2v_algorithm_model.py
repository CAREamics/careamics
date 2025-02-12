""""N2V Algorithm configuration."""

from typing import Annotated, Literal

from pydantic import AfterValidator, ConfigDict, model_validator
from typing_extensions import Self

from careamics.config.architectures import UNetModel
from careamics.config.support import SupportedPixelManipulation, SupportedStructAxis
from careamics.config.transformations import N2VManipulateModel
from careamics.config.validators import (
    model_matching_in_out_channels,
    model_without_final_activation,
)

from .unet_algorithm_model import UNetBasedAlgorithm


class N2VAlgorithm(UNetBasedAlgorithm):
    """N2V Algorithm configuration."""

    model_config = ConfigDict(validate_assignment=True)

    algorithm: Literal["n2v"] = "n2v"
    """N2V Algorithm name."""

    loss: Literal["n2v"] = "n2v"
    """N2V loss function."""

    n2v_config: N2VManipulateModel = N2VManipulateModel()

    model: Annotated[
        UNetModel,
        AfterValidator(model_matching_in_out_channels),
        AfterValidator(model_without_final_activation),
    ]

    @model_validator(mode="after")
    def validate_n2v2(self) -> Self:
        """Validate that the N2V2 strategy and models are set correctly.

        Returns
        -------
        Self
            The validateed configuration.

        Raises
        ------
        ValueError
            If N2V2 is used with the wrong pixel manipulation strategy.
        """
        if self.model.n2v2:
            if self.n2v_config.strategy != SupportedPixelManipulation.MEDIAN.value:
                raise ValueError(
                    f"N2V2 can only be used with the "
                    f"{SupportedPixelManipulation.MEDIAN} pixel manipulation strategy. "
                    f"Change the `strategy` parameters in `n2v_config` to "
                    f"{SupportedPixelManipulation.MEDIAN}."
                )
        else:
            if self.n2v_config.strategy != SupportedPixelManipulation.UNIFORM.value:
                raise ValueError(
                    f"N2V can only be used with the "
                    f"{SupportedPixelManipulation.UNIFORM} pixel manipulation strategy."
                    f" Change the `strategy` parameters in `n2v_config` to "
                    f"{SupportedPixelManipulation.UNIFORM}."
                )
        return self

    def set_n2v2(self, use_n2v2: bool) -> None:
        """
        Set the configuration to use N2V2 or the vanilla Noise2Void.

        This method ensures that N2V2 is set correctly and remain coherent, as opposed
        to setting the different parameters individually.

        Parameters
        ----------
        use_n2v2 : bool
            Whether to use N2V2.
        """
        if use_n2v2:
            self.n2v_config.strategy = SupportedPixelManipulation.MEDIAN.value
            self.model.n2v2 = True
        else:
            self.n2v_config.strategy = SupportedPixelManipulation.UNIFORM.value
            self.model.n2v2 = False

    def is_struct_n2v(self) -> bool:
        """Check if the configuration is using structN2V.

        Returns
        -------
        bool
            Whether the configuration is using structN2V.
        """
        return self.n2v_config.struct_mask_axis != SupportedStructAxis.NONE.value
