"""HDN Pydantic configuration."""

from bioimageio.spec.generic.v0_3 import CiteEntry

from careamics.config.algorithms.hdn_algorithm_model import HDNAlgorithm
from careamics.config.configuration import Configuration
from careamics.config.data import DataConfig

HDN = "HDN"

HDN_DESCRIPTION = (
    ""
)
HDN_REF = CiteEntry(
    text='',
    doi="",
)


class HDNConfiguration(Configuration):
    """HDN configuration."""

    algorithm_config: HDNAlgorithm
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
            "UNet",
            "3D" if "Z" in self.data_config.axes else "2D",
            "HDNamics",
            "pytorch",
            HDN,
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