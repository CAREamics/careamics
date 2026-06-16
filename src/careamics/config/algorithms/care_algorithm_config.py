"""CARE algorithm configuration."""

from typing import Annotated, Literal

from bioimageio.spec.generic.v0_3 import CiteEntry
from pydantic import AfterValidator

from careamics.config.architectures import UNetConfig
from careamics.config.validators import (
    model_no_c_ind_for_mismatching_channels,
    model_without_final_activation,
    model_without_n2v2,
)
from careamics.references.care import (
    CARE,
    CARE_DESCRIPTION,
    CARE_REF,
)

from .unet_algorithm_config import UNetBasedAlgorithm


class CAREAlgorithm(UNetBasedAlgorithm):
    """CARE algorithm configuration.

    Attributes
    ----------
    algorithm : "care"
        CARE Algorithm name.
    loss : {"mae", "mse"}
        CARE-compatible loss function.
    """

    algorithm: Literal["care"] = "care"
    """CARE Algorithm name."""

    loss: Literal["mae", "mse"] = "mae"
    """CARE-compatible loss function."""

    model: Annotated[
        UNetConfig,
        AfterValidator(model_without_n2v2),
        AfterValidator(model_without_final_activation),
        AfterValidator(model_no_c_ind_for_mismatching_channels),
    ]
    """UNet without a final activation function, without the `n2v2` modifications, and
    without independent channels for mismatching input/output channel numbers."""

    def get_algorithm_friendly_name(self) -> str:
        """
        Get the algorithm friendly name.

        Returns
        -------
        str
            Friendly name of the algorithm.
        """
        return CARE

    def get_algorithm_keywords(self) -> list[str]:
        """
        Get algorithm keywords.

        Returns
        -------
        list[str]
            List of keywords.
        """
        return [
            "restoration",
            "UNet",
            "3D" if self.model.is_3D() else "2D",
            "CAREamics",
            "pytorch",
            CARE,
        ]

    def get_algorithm_references(self) -> str:
        """
        Get the algorithm references.

        This is used to generate the README of the BioImage Model Zoo export.

        Returns
        -------
        str
            Algorithm references.
        """
        return CARE_REF.text + " doi: " + str(CARE_REF.doi)

    def get_algorithm_citations(self) -> list[CiteEntry]:
        """
        Return a list of citation entries of the current algorithm.

        This is used to generate the model description for the BioImage Model Zoo.

        Returns
        -------
        List[CiteEntry]
            List of citation entries.
        """
        return [CARE_REF]

    def get_algorithm_description(self) -> str:
        """
        Get the algorithm description.

        Returns
        -------
        str
            Algorithm description.
        """
        return CARE_DESCRIPTION

    @classmethod
    def is_supervised(cls) -> bool:
        """
        Return whether the algorithm is supervised.

        Returns
        -------
        bool
            Whether the algorithm is supervised.
        """
        return True
