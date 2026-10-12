"""N2N Algorithm configuration."""

from typing import Annotated, Literal

from bioimageio.spec.generic.v0_3 import CiteEntry
from pydantic import AfterValidator

from careamics.config.architectures import UNetConfig
from careamics.config.validators import (
    model_no_c_ind_for_mismatching_channels,
    model_without_final_activation,
    model_without_n2v2,
)
from careamics.references.n2n import (
    N2N,
    N2N_DESCRIPTION,
    N2N_REF,
)

from .unet_algorithm_config import UNetBasedAlgorithm


class N2NAlgorithm(UNetBasedAlgorithm):
    """Noise2Noise Algorithm configuration."""

    algorithm: Literal["n2n"] = "n2n"
    """N2N Algorithm name."""

    loss: Literal["mae", "mse"] = "mae"
    """N2N-compatible loss function."""

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
        return N2N

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
            N2N,
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
        return N2N_REF.text + " doi: " + str(N2N_REF.doi)

    def get_algorithm_citations(self) -> list[CiteEntry]:
        """
        Return a list of citation entries of the current algorithm.

        This is used to generate the model description for the BioImage Model Zoo.

        Returns
        -------
        List[CiteEntry]
            List of citation entries.
        """
        return [N2N_REF]

    def get_algorithm_description(self) -> str:
        """
        Get the algorithm description.

        Returns
        -------
        str
            Algorithm description.
        """
        return N2N_DESCRIPTION

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
