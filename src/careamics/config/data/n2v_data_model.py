"""Noise2Void specific data configuration model."""

from collections.abc import Sequence
from typing import Literal

from pydantic import Field, field_validator

from careamics.config.data.data_model import GeneralDataConfig
from careamics.config.support import SupportedTransform
from careamics.config.transformations import (
    N2V_TRANSFORMS_UNION,
    N2VManipulateModel,
    XYFlipModel,
    XYRandomRotate90Model,
)


class N2VDataConfig(GeneralDataConfig):
    """N2V specific data configuration model."""

    transforms: Sequence[N2V_TRANSFORMS_UNION] = Field(
        default=[XYFlipModel(), XYRandomRotate90Model(), N2VManipulateModel()],
        validate_default=True,
    )
    """N2V compatible transforms. N2VManpulate should be the last transform."""

    @field_validator("transforms")
    @classmethod
    def validate_transforms(
        cls, transforms: list[N2V_TRANSFORMS_UNION]
    ) -> list[N2V_TRANSFORMS_UNION]:
        """
        Validate N2VManipulate transform position in the transform list.

        Parameters
        ----------
        transforms : list of transforms compatible with N2V
            Transforms.

        Returns
        -------
        list of transforms
            Validated transforms.

        Raises
        ------
        ValueError
            If multiple instances of N2VManipulate are found or if it is not the last
            transform.
        """
        transform_list = [t.name for t in transforms]

        if SupportedTransform.N2V_MANIPULATE in transform_list:
            # multiple N2V_MANIPULATE
            if transform_list.count(SupportedTransform.N2V_MANIPULATE.value) > 1:
                raise ValueError(
                    f"Multiple instances of "
                    f"{SupportedTransform.N2V_MANIPULATE} transforms "
                    f"are not allowed."
                )

            # N2V_MANIPULATE not the last transform
            elif transform_list[-1] != SupportedTransform.N2V_MANIPULATE:
                raise ValueError(
                    f"{SupportedTransform.N2V_MANIPULATE} transform "
                    f"should be the last transform."
                )

        else:
            raise ValueError(
                f"{SupportedTransform.N2V_MANIPULATE} transform "
                f"is required for N2V training."
            )

        return transforms

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
        found_n2v = False

        for transform in self.transforms:
            if transform.name == SupportedTransform.N2V_MANIPULATE.value:
                transform.strategy = strategy
                found_n2v = True

        if not found_n2v:
            transforms = [t.name for t in self.transforms]
            raise ValueError(
                f"N2V_Manipulate transform not found in the transforms list "
                f"({transforms})."
            )

    def get_masking_strategy(self) -> Literal["uniform", "median"]:
        """
        Get N2V2 strategy.

        Returns
        -------
        "uniform" or "median"
            Strategy used for N2V2.
        """
        for transform in self.transforms:
            if transform.name == SupportedTransform.N2V_MANIPULATE.value:
                return transform.strategy

        raise ValueError(
            f"{SupportedTransform.N2V_MANIPULATE} transform "
            f"is required for N2V training."
        )

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
        found_n2v = False

        for transform in self.transforms:
            if transform.name == SupportedTransform.N2V_MANIPULATE.value:
                transform.struct_mask_axis = mask_axis
                transform.struct_mask_span = mask_span
                found_n2v = True

        if not found_n2v:
            transforms = [t.name for t in self.transforms]
            raise ValueError(
                f"N2V pixel manipulate transform not found in the transforms "
                f"({transforms})."
            )

    def is_using_struct_n2v(self) -> bool:
        """
        Check if structN2V is enabled.

        Returns
        -------
        bool
            Whether structN2V is enabled or not.
        """
        for transform in self.transforms:
            if transform.name == SupportedTransform.N2V_MANIPULATE.value:
                return transform.struct_mask_axis != "none"

        raise ValueError(
            f"N2V pixel manipulate transform not found in the transforms "
            f"({self.transforms})."
        )
