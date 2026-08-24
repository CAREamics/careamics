"""Segmentation with UNet algorithm configuration."""

from typing import Annotated, Literal

from bioimageio.spec.generic.v0_3 import CiteEntry
from pydantic import AfterValidator

from careamics.config.algorithms.unet_algorithm_config import UNetBasedAlgorithm
from careamics.config.architectures import UNetConfig
from careamics.config.validators import (
    model_without_final_activation,
    model_without_n2v2,
)


def _model_with_at_least_2_classes(model: UNetConfig) -> UNetConfig:
    """Validate that the Unet model has at least two classes.

    Parameters
    ----------
    model : UNetConfig
        Model to validate.

    Returns
    -------
    UNetConfig
        The validated model.

    Raises
    ------
    ValueError
        If the model has less than two classes.
    """
    if model.num_classes < 2:
        raise ValueError(
            f"U-Net model should have at least 2 classes, including background, "
            f"got {model.num_classes} classes."
        )

    return model


def _model_with_dependent_channels(model: UNetConfig) -> UNetConfig:
    """Validate that the Unet model has dependent channels.

    Parameters
    ----------
    model : UNetConfig
        Model to validate.

    Returns
    -------
    UNetConfig
        The validated model.

    Raises
    ------
    ValueError
        If the model has independent channels.
    """
    if model.independent_channels:
        raise ValueError(
            "U-Net model should have `independent_channels` set to `False`."
        )

    return model


class SegAlgorithm(UNetBasedAlgorithm):
    """Configuration for segmentation algorithm."""

    algorithm: Literal["seg"] = "seg"
    """Segmentation algorithm name."""

    loss: Literal["dice", "ce", "dice_ce"] = "dice"
    """Segmentation-compatible loss function."""

    model: Annotated[
        UNetConfig,
        AfterValidator(model_without_n2v2),
        AfterValidator(model_without_final_activation),
        AfterValidator(_model_with_at_least_2_classes),
        AfterValidator(_model_with_dependent_channels),
    ]
    """UNet without a final activation function and without the `n2v2` modifications."""

    def get_algorithm_friendly_name(self) -> str:
        """
        Get the friendly name of the algorithm.

        Returns
        -------
        str
            Friendly name.
        """
        return "UNet semantic segmentation"

    def get_algorithm_keywords(self) -> list[str]:
        """
        Get algorithm keywords.

        Returns
        -------
        list[str]
            List of keywords.
        """
        keywords = [
            "semantic segmentation",
            "UNet",
            "3D" if self.model.is_3D() else "2D",
            "CAREamics",
            "pytorch",
        ]

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
        return ""

    def get_algorithm_citations(self) -> list[CiteEntry]:
        """
        Return a list of citation entries of the current algorithm.

        This is used to generate the model description for the BioImage Model Zoo.

        Returns
        -------
        List[CiteEntry]
            List of citation entries.
        """
        return []

    def get_algorithm_description(self) -> str:
        """
        Return a description of the algorithm.

        This method is used to generate the README of the BioImage Model Zoo export.

        Returns
        -------
        str
            Description of the algorithm.
        """
        return "UNet semantic segmentation."

    @classmethod
    def is_supervised(cls) -> bool:
        """
        Whether the algorithm is supervised.

        Returns
        -------
        bool
            Whether the algorithm is supervised.
        """
        return True
