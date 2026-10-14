"""COSDD algorithm configuration."""

from pprint import pformat
from typing import Literal

from bioimageio.spec.generic.v0_3 import CiteEntry
from pydantic import BaseModel, ConfigDict, Field

from careamics.config.architectures import COSDDConfig
from careamics.config.lightning.optimizer_configs import (
    LrSchedulerConfig,
    OptimizerConfig,
)
from careamics.references.cosdd import COSDD, COSDD_DESCRIPTION, COSDD_REF


class COSDDAlgorithm(BaseModel):
    """COSDD algorithm configuration.

    COSDD trains three sets of networks jointly, each with its own optimizer and
    learning rate scheduler built from `optimizer` and `lr_scheduler`: the Ladder VAE
    with the autoregressive noise decoder, the signal decoder and, optionally, the
    direct denoiser.
    """

    model_config = ConfigDict(
        protected_namespaces=(),  # allows to use model_* as a field name
        validate_assignment=True,
    )

    algorithm: Literal["cosdd"] = "cosdd"
    """COSDD algorithm name."""

    model: COSDDConfig
    """COSDD networks configuration."""

    optimizer: OptimizerConfig = OptimizerConfig(name="Adamax")
    """Optimizer used for each of the COSDD networks."""

    lr_scheduler: LrSchedulerConfig = LrSchedulerConfig(
        name="ReduceLROnPlateau", parameters={"factor": 0.1, "patience": 25}
    )
    """Learning rate scheduler used for each of the COSDD networks. A
    `ReduceLROnPlateau` scheduler monitors the validation loss of the network it
    schedules."""

    n_grad_batches: int = Field(default=4, ge=1)
    """Number of batches over which gradients are accumulated before updating the
    weights. COSDD uses manual optimization, so the Lightning Trainer
    `accumulate_grad_batches` parameter is not supported."""

    def __str__(self) -> str:
        """Pretty string representing the configuration.

        Returns
        -------
        str
            Pretty string.
        """
        return pformat(self.model_dump())

    def get_algorithm_friendly_name(self) -> str:
        """
        Get the algorithm friendly name.

        Returns
        -------
        str
            Friendly name of the algorithm.
        """
        return COSDD

    def get_algorithm_keywords(self) -> list[str]:
        """
        Get algorithm keywords.

        Returns
        -------
        list[str]
            List of keywords.
        """
        return [
            "denoising",
            "restoration",
            "VAE",
            "3D" if self.model.is_3D() else "2D",
            "CAREamics",
            "pytorch",
            COSDD,
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
        return COSDD_REF.text + " doi: " + str(COSDD_REF.doi)

    def get_algorithm_citations(self) -> list[CiteEntry]:
        """
        Return a list of citation entries of the current algorithm.

        This is used to generate the model description for the BioImage Model Zoo.

        Returns
        -------
        List[CiteEntry]
            List of citation entries.
        """
        return [COSDD_REF]

    def get_algorithm_description(self) -> str:
        """
        Get the algorithm description.

        Returns
        -------
        str
            Algorithm description.
        """
        return COSDD_DESCRIPTION

    @classmethod
    def is_supervised(cls) -> bool:
        """
        Return whether the algorithm is supervised.

        Returns
        -------
        bool
            Whether the algorithm is supervised.
        """
        return False
