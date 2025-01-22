"""N2N configuration."""

from bioimageio.spec.generic.v0_3 import CiteEntry

from careamics.config.algorithms import N2NAlgorithm
from careamics.config.configuration import Configuration
from careamics.config.data import DataConfig

N2N = "Noise2Noise"

N2N_DESCRIPTION = (
    "Noise2Noise is a deep-learning-based algorithm that uses a U-Net "
    "architecture to restore images. Noise2Noise is a self-supervised "
    "algorithm that requires only noisy images to train the network. "
    "The algorithm learns to predict the clean image from the noisy "
    "image. Noise2Noise is particularly useful when clean images are "
    "not available for training."
)

N2N_REF = CiteEntry(
    text="Lehtinen, J., Munkberg, J., Hasselgren, J., Laine, S., Karras, T., "
    'Aittala, M. and Aila, T., 2018. "Noise2Noise: Learning image restoration '
    'without clean data". arXiv preprint arXiv:1803.04189.',
    doi="10.48550/arXiv.1803.04189",
)


class N2NConfiguration(Configuration):
    """Noise2Noise configuration."""

    algorithm_config: N2NAlgorithm
    """Algorithm configuration."""

    data_config: DataConfig
    """Data configuration."""

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
            "3D" if "Z" in self.data_config.axes else "2D",
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
        return N2N_REF.text + " doi: " + N2N_REF.doi

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
