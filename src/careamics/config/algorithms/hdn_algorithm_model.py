"""HDN algorithm configuration."""

from typing import Literal

from bioimageio.spec.generic.v0_3 import CiteEntry
from pydantic import ConfigDict

from careamics.config.algorithms.vae_algorithm_model import VAEBasedAlgorithm
from careamics.config.architectures import LVAEModel
from careamics.config.loss_model import LVAELossConfig

HDN = "Hierarchical DivNoising"

HDN_DESCRIPTION = (
    "HDN leverages a hierarchical VAE to perform image "
    "restoration. It is designed to be interpretable and unsupervised, "
    "making it suitable for a wide range of microscopy images."
)
HDN_REF = CiteEntry(
    text='Prakash, M., Delbracio, M., Milanfar, P., Jug, F. 2022. "Interpretable '
    'Unsupervised Diversity Denoising and Artefact Removal." The International '
    "Conference on Learning Representations (ICLR).",
    doi="10.1561/2200000056",
)


class HDNAlgorithm(VAEBasedAlgorithm):
    """HDN algorithm configuration."""

    model_config = ConfigDict(validate_assignment=True)

    algorithm: Literal["hdn"] = "hdn"

    loss: LVAELossConfig

    model: LVAEModel  # TODO add validators

    is_supervised: bool = False

    def get_algorithm_friendly_name(self) -> str:
        """
        Get the algorithm friendly name.

        Returns
        -------
        str
            Friendly name of the algorithm.
        """
        return HDN

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
        return HDN_REF.text + " doi: " + HDN_REF.doi

    def get_algorithm_citations(self) -> list[CiteEntry]:
        """
        Return a list of citation entries of the current algorithm.

        This is used to generate the model description for the BioImage Model Zoo.

        Returns
        -------
        List[CiteEntry]
            List of citation entries.
        """
        return [HDN_REF]

    def get_algorithm_description(self) -> str:
        """
        Get the algorithm description.

        Returns
        -------
        str
            Algorithm description.
        """
        return HDN_DESCRIPTION
