"""MicroSplit algorithm configuration."""

from typing import Literal

from bioimageio.spec.generic.v0_3 import CiteEntry
from pydantic import ConfigDict

from careamics.config.algorithms.vae_algorithm_model import VAEBasedAlgorithm
from careamics.config.architectures import LVAEModel
from careamics.config.loss_model import LVAELossConfig

MICROSPLIT = "MicroSplit"

MICROSPLIT_DESCRIPTION = """MicroSplit is a self-supervised deep learning method for
microscopy image splitting that combines the strengths of both denoising and
representation learning approaches."""

MICROSPLIT_REF = CiteEntry(
    text='Prakash, M., Delbracio, M., Milanfar, P., Jug, F. 2022. "Interpretable '
    'Unsupervised Diversity Denoising and Artefact Removal." The International '
    "Conference on Learning Representations (ICLR).",
    doi="10.1561/2200000056",
)


class MicroSplitAlgorithm(VAEBasedAlgorithm):
    """MicroSplit algorithm configuration."""

    model_config = ConfigDict(validate_assignment=True)

    algorithm: Literal["microsplit"] = "microsplit"

    loss: LVAELossConfig

    model: LVAEModel  # TODO add validators

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
