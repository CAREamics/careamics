"""N2V Algorithm configuration."""

from typing import Annotated, Literal, Self

from bioimageio.spec.generic.v0_3 import CiteEntry
from pydantic import AfterValidator, ConfigDict, model_validator

from careamics.config.architectures import UNetConfig
from careamics.config.support import SupportedPixelManipulation
from careamics.config.validators import (
    model_matching_in_out_channels,
    model_without_final_activation,
)
from careamics.references.n2v import (
    N2V,
    N2V2,
    N2V2_DESCRIPTION,
    N2V2_REF,
    N2V_DESCRIPTION,
    N2V_REF,
    STR_N2V2_DESCRIPTION,
    STR_N2V_DESCRIPTION,
    STRUCT_N2V,
    STRUCT_N2V2,
    STRUCTN2V_REF,
)

from .n2v_manipulation import N2VManipulateConfig
from .unet_algorithm_config import UNetBasedAlgorithm


class N2VAlgorithm(UNetBasedAlgorithm):
    """N2V Algorithm configuration."""

    model_config = ConfigDict(
        validate_assignment=True,
    )

    algorithm: Literal["n2v"] = "n2v"
    """N2V Algorithm name."""

    loss: Literal["n2v"] = "n2v"
    """N2V loss function."""

    monitor_metric: Literal["train_loss", "train_loss_epoch", "val_loss"] = "val_loss"
    """Metric to monitor for the learning rate scheduler. Used in the returned dict of
    PyTorch Lightning `configure_optimizers` method."""

    n2v_config: N2VManipulateConfig = N2VManipulateConfig()
    """Noise2Void pixel manipulation configuration."""

    model: Annotated[
        UNetConfig,
        AfterValidator(model_matching_in_out_channels),
        AfterValidator(model_without_final_activation),
    ]
    """Model parameters."""

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
        return self.n2v_config.struct_mask is not None

    def get_algorithm_friendly_name(self) -> str:
        """
        Get the friendly name of the algorithm.

        Returns
        -------
        str
            Friendly name.
        """
        use_n2v2 = self.model.n2v2
        use_structN2V = self.is_struct_n2v()

        if use_n2v2 and use_structN2V:
            return STRUCT_N2V2
        elif use_n2v2:
            return N2V2
        elif use_structN2V:
            return STRUCT_N2V
        else:
            return N2V

    def get_algorithm_keywords(self) -> list[str]:
        """
        Get algorithm keywords.

        Returns
        -------
        list[str]
            List of keywords.
        """
        use_n2v2 = self.model.n2v2
        use_structN2V = self.is_struct_n2v()

        keywords = [
            "denoising",
            "restoration",
            "UNet",
            "3D" if self.model.is_3D() else "2D",
            "CAREamics",
            "pytorch",
            N2V,
        ]

        if use_n2v2:
            keywords.append(N2V2)
        if use_structN2V:
            keywords.append(STRUCT_N2V)

        return keywords

    def get_algorithm_references(self) -> str:
        """
        Get the algorithm references.

        This is used to generate the README of the BioImage Model Zoo export.

        Returns
        -------
        str
            Algorithm references.
        """
        use_n2v2 = self.model.n2v2
        use_structN2V = self.is_struct_n2v()

        references = [
            N2V_REF.text + " doi: " + str(N2V_REF.doi),
            N2V2_REF.text + " doi: " + str(N2V2_REF.doi),
            STRUCTN2V_REF.text + " doi: " + str(STRUCTN2V_REF.doi),
        ]

        # return the (struct)N2V(2) references
        if use_n2v2 and use_structN2V:
            return "\n".join(references)
        elif use_n2v2:
            references.pop(-1)
            return "\n".join(references)
        elif use_structN2V:
            references.pop(-2)
            return "\n".join(references)
        else:
            return references[0]

    def get_algorithm_citations(self) -> list[CiteEntry]:
        """
        Return a list of citation entries of the current algorithm.

        This is used to generate the model description for the BioImage Model Zoo.

        Returns
        -------
        List[CiteEntry]
            List of citation entries.
        """
        use_n2v2 = self.model.n2v2
        use_structN2V = self.is_struct_n2v()

        references = [N2V_REF]

        if use_n2v2:
            references.append(N2V2_REF)

        if use_structN2V:
            references.append(STRUCTN2V_REF)

        return references

    def get_algorithm_description(self) -> str:
        """
        Return a description of the algorithm.

        This method is used to generate the README of the BioImage Model Zoo export.

        Returns
        -------
        str
            Description of the algorithm.
        """
        use_n2v2 = self.model.n2v2
        use_structN2V = self.is_struct_n2v()

        if use_n2v2 and use_structN2V:
            return STR_N2V2_DESCRIPTION
        elif use_n2v2:
            return N2V2_DESCRIPTION
        elif use_structN2V:
            return STR_N2V_DESCRIPTION
        else:
            return N2V_DESCRIPTION

    @classmethod
    def is_supervised(cls) -> bool:
        """
        Return whether the algorithm is supervised.

        Returns
        -------
        bool
            Whether the algorithm is supervised.
        """
        return False
