"""MicroSplit algorithm configuration."""

from typing import Literal

from bioimageio.spec.generic.v0_3 import CiteEntry
from pydantic import ConfigDict

from careamics.config.algorithms.vae_algorithm_config import VAEBasedAlgorithm
from careamics.config.architectures import LVAEConfig
from careamics.config.losses.loss_config import LVAELossConfig

MICROSPLIT = "MicroSplit"

MICROSPLIT_DESCRIPTION = (
    "A computational multiplexing technique based on deep learning that allows multiple"
    " cellular structures to be imaged in a single fluorescent channel and then unmix "
    "them by computational means, allowing faster imaging and reduced photon exposure."
)

MICROSPLIT_REF = CiteEntry(
    text=(
        'Ashesh, et al. "MicroSplit: Semantic Unmixing of Fluorescent Microscopy '
        'Data.", Nat. Methods (2026).'
    ),
    doi="10.1101/2025.02.10.637323 ",
)


class MicroSplitAlgorithm(VAEBasedAlgorithm):
    """MicroSplit algorithm configuration."""

    model_config = ConfigDict(validate_assignment=True)

    algorithm: Literal["microsplit"] = "microsplit"

    loss: LVAELossConfig

    model: LVAEConfig  # TODO add validators

    is_supervised: bool = True

    def get_algorithm_friendly_name(self) -> str:
        """
        Get the algorithm friendly name.

        Returns
        -------
        str
            Friendly name of the algorithm.
        """
        return MICROSPLIT

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
            "VAE",
            "self-supervised",
            "3D" if self.model.is_3D() else "2D",
            "CAREamics",
            "pytorch",
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
        return MICROSPLIT_REF.text + " doi: " + MICROSPLIT_REF.doi

    def get_algorithm_citations(self) -> list[CiteEntry]:
        """
        Return a list of citation entries of the current algorithm.

        This is used to generate the model description for the BioImage Model Zoo.

        Returns
        -------
        List[CiteEntry]
            List of citation entries.
        """
        return [MICROSPLIT_REF]

    def get_algorithm_description(self) -> str:
        """
        Get the algorithm description.

        Returns
        -------
        str
            Algorithm description.
        """
        return MICROSPLIT_DESCRIPTION
