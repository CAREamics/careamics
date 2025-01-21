""""N2V Algorithm configuration."""

from typing import Annotated, Literal

from pydantic import AfterValidator, ConfigDict

from careamics.config.architectures import UNetModel
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

    n2v_masking: N2VManipulateModel = N2VManipulateModel()

    model: Annotated[
        UNetModel,
        AfterValidator(model_matching_in_out_channels),
        AfterValidator(model_without_final_activation),
    ]

    def get_masking_strategy(self) -> str:
        """Get the masking strategy for N2V."""
        return self.n2v_masking.strategy

    def set_masking_strategy(self, strategy: Literal["uniform", "median"]) -> None:
        """
        Set masking strategy.

        Parameters
        ----------
        strategy : "uniform" or "median"
            Strategy to use for N2V2.

        Raises
        ------
        ValueError
            If the N2V pixel manipulate transform is not found in the transforms.
        """
        self.model.n2v_masking.strategy = strategy

    def set_n2v2(self, use_n2v2: bool) -> None:
        """
        Set the configuration to use N2V2 or the vanilla Noise2Void.

        Parameters
        ----------
        use_n2v2 : bool
            Whether to use N2V2.
        """
        if use_n2v2:
            self.set_masking_strategy("median")
        else:
            self.set_masking_strategy("uniform")

    def is_using_struct_n2v(self) -> bool:
        """Check if the configuration is using structN2V."""
        return self.n2v_masking.struct_mask_axis != "none"  # TODO change!

    def set_structN2V_mask(
        self, mask_axis: Literal["horizontal", "vertical", "none"], mask_span: int
    ) -> None:
        """
        Set structN2V mask parameters.

        Setting `mask_axis` to `none` will disable structN2V.

        Parameters
        ----------
        mask_axis : Literal["horizontal", "vertical", "none"]
            Axis along which to apply the mask. `none` will disable structN2V.
        mask_span : int
            Total span of the mask in pixels.

        Raises
        ------
        ValueError
            If the N2V pixel manipulate transform is not found in the transforms.
        """
        self.n2v_masking.struct_mask_axis = mask_axis
        self.n2v_masking.struct_mask_span = mask_span
