"""CARE algorithm configuration."""

from typing import Annotated, Literal

from bioimageio.spec.generic.v0_3 import CiteEntry
from pydantic import AfterValidator

from careamics.config.architectures import UNetModel
from careamics.config.validators import (
    model_without_final_activation,
    model_without_n2v2,
)

from .unet_algorithm_model import UNetBasedAlgorithm

CARE = "CARE"

CARE_DESCRIPTION = (
    "Content-aware image restoration (CARE) is a deep-learning-based "
    "algorithm that uses a U-Net architecture to restore images. CARE "
    "is a supervised algorithm that requires pairs of noisy and "
    "clean images to train the network. The algorithm learns to "
    "predict the clean image from the noisy image. CARE is "
    "particularly useful for denoising images acquired in low-light "
    "conditions, such as fluorescence microscopy images."
)
CARE_REF = CiteEntry(
    text='Weigert, Martin, et al. "Content-aware image restoration: pushing the '
    'limits of fluorescence microscopy." Nature methods 15.12 (2018): 1090-1097.',
    doi="10.1038/s41592-018-0216-7",
)


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
        UNetModel,
        AfterValidator(model_without_n2v2),
        AfterValidator(model_without_final_activation),
    ]
    """UNet without a final activation function and without the `n2v2` modifications."""

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
        return CARE_REF.text + " doi: " + CARE_REF.doi

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
