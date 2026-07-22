"""HDN algorithm configuration."""

from pprint import pformat
from typing import Literal, Self

from bioimageio.spec.generic.v0_3 import CiteEntry
from pydantic import BaseModel, ConfigDict, model_validator

from careamics.config.architectures import LVAEConfig
from careamics.config.lightning.optimizer_configs import (
    LrSchedulerConfig,
    OptimizerConfig,
)
from careamics.config.losses.loss_config import LVAELossConfig
from careamics.config.noise_model.noise_model_config import MultiChannelNMConfig
from careamics.config.support import SupportedLoss
from careamics.config.validators import (
    noise_models_match_output_channels,
    predict_logvar_consistent,
)

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


class HDNAlgorithm(BaseModel):
    """HDN algorithm configuration."""

    model_config = ConfigDict(
        protected_namespaces=(),  # allows to use model_* as a field name
        validate_assignment=True,
    )

    algorithm: Literal["hdn"] = "hdn"

    loss: LVAELossConfig = LVAELossConfig(loss_type="hdn")

    model: LVAEConfig

    noise_model: MultiChannelNMConfig | None = None

    mmse_count: int = 1

    is_supervised: bool = False

    # overwrite default optimizer
    optimizer: OptimizerConfig = OptimizerConfig(name="Adamax")
    """Optimizer to use, defined in SupportedOptimizer."""

    lr_scheduler: LrSchedulerConfig = LrSchedulerConfig()

    @model_validator(mode="after")
    def validate_constraints(self: Self) -> Self:
        """Validate the algorithm-specific constraints.

        Returns
        -------
        Self
            The validated model.

        Raises
        ------
        ValueError
            If the loss, model or noise model configurations are not compatible
            with the HDN algorithm.
        """
        if self.loss.loss_type != SupportedLoss.HDN:
            raise ValueError(f"HDN only supports loss `hdn`.")

        if self.model.multiscale_count > 1:
            raise ValueError("Algorithm `hdn` does not support multiscale models.")

        if self.model.output_channels != 1:
            raise ValueError(
                f"Number of output channels ({self.model.output_channels}) must be 1 "
                "for algorithm `hdn`."
            )

        predict_logvar_consistent(self.model, self.loss)
        noise_models_match_output_channels(self.model, self.noise_model)

        return self

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
