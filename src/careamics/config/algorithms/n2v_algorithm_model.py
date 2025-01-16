""""N2V Algorithm configuration."""

from typing import Literal

from pydantic import ConfigDict, model_validator
from typing_extensions import Self

from careamics.config.transformations import N2VManipulateModel

from .unet_algorithm_model import UNetBasedAlgorithm


class N2VAlgorithm(UNetBasedAlgorithm):
    """N2V Algorithm configuration."""

    model_config = ConfigDict(validate_assignment=True)

    algorithm: Literal["n2v"] = "n2v"
    """N2V Algorithm name."""

    loss: Literal["n2v"] = "n2v"
    """N2V loss function."""

    n2v_masking: N2VManipulateModel = N2VManipulateModel()

    @model_validator(mode="after")
    def algorithm_cross_validation(self: Self) -> Self:
        """Validate the algorithm model for N2V.

        Returns
        -------
        Self
            The validated model.
        """
        if self.model.in_channels != self.model.num_classes:
            raise ValueError(
                "N2V requires the same number of input and output channels. Make "
                "sure that `in_channels` and `num_classes` are the same."
            )

        if self.model.n2v2 is True:
            self.set_masking_strategy("median")

        return self

    def set_n2v2(self, use_n2v2: bool) -> None:
        """
        Set the N2V transform to the N2V2 version.

        Parameters
        ----------
        use_n2v2 : bool
            Whether to use N2V2.

        Raises
        ------
        ValueError
            If the N2V pixel manipulate transform is not found in the transforms.
        """
        if use_n2v2:
            self.set_masking_strategy("median")
        else:
            self.set_masking_strategy("uniform")

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
        self.n2v_masking.strategy = strategy

    def get_masking_strategy(self) -> Literal["uniform", "median"]:
        """
        Get N2V2 strategy.

        Returns
        -------
        "uniform" or "median"
            Strategy used for N2V2.
        """
        return self.n2v_masking.strategy

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


    def is_using_struct_n2v(self) -> bool:
        """
        Check if structN2V is enabled.

        Returns
        -------
        bool
            Whether structN2V is enabled or not.
        """
        return self.n2v_masking.struct_mask_axis != "none"

