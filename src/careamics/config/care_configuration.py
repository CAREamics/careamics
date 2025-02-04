"""CARE Pydantic configuration."""

from bioimageio.spec.generic.v0_3 import CiteEntry

from careamics.config.algorithms.care_algorithm_model import CAREAlgorithm
from careamics.config.configuration import Configuration
from careamics.config.data import DataConfig

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


class CAREConfiguration(Configuration):
    """CARE configuration."""

    algorithm_config: CAREAlgorithm
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
            "3D" if "Z" in self.data_config.axes else "2D",
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
