"""MicroSplit algorithm configuration."""

import warnings
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


class MicroSplitAlgorithm(BaseModel):
    """MicroSplit algorithm configuration."""

    model_config = ConfigDict(
        protected_namespaces=(),  # allows to use model_* as a field name
        validate_assignment=True,
    )

    algorithm: Literal["microsplit"] = "microsplit"

    loss: LVAELossConfig = LVAELossConfig(loss_type="microsplit")

    model: LVAEConfig

    noise_model: MultiChannelNMConfig | None = None

    mmse_count: int = 1

    is_supervised: bool = True

    optimizer: OptimizerConfig = OptimizerConfig()
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
            with the MicroSplit algorithm.
        """
        if self.loss.loss_type != SupportedLoss.MICROSPLIT:
            raise ValueError(
                f"Algorithm {self.algorithm} only supports loss `microsplit`."
            )

        # Remind users to attach a noise model when using denoiSplit
        if self.loss.denoisplit_weight > 0 and self.noise_model is None:
            warnings.warn(
                "denoisplit_weight > 0 but no noise_model is provided in the "
                "configuration. A noise model is required for denoiSplit training. "
                "Train one with NoiseModelTrainer, then pass the result to "
                "create_microsplit_config(noise_model=trainer.get_config()) "
                "before training.",
                UserWarning,
                stacklevel=2,
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
