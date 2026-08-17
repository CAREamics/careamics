"""MicroSplit algorithm configuration."""

import warnings
from pprint import pformat
from typing import Annotated, Literal, Self

from bioimageio.spec.generic.v0_3 import CiteEntry
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from careamics.config.architectures import LVAEConfig
from careamics.config.lightning.optimizer_configs import (
    LrSchedulerConfig,
    OptimizerConfig,
)
from careamics.config.losses.loss_config import MicroSplitLossConfig
from careamics.config.noise_model.noise_model_config import MultiChannelNMConfig
from careamics.config.validators import (
    at_least_one_likelihood,
    lvae_conv_strides_valid,
    lvae_depth_valid,
    lvae_multiscale_count_valid,
    lvae_spatial_shape_valid,
    noise_models_match_output_channels,
    predict_logvar_consistent,
    predict_logvar_required_for_musplit,
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

    loss: Annotated[
        MicroSplitLossConfig,
        AfterValidator(at_least_one_likelihood),
        AfterValidator(predict_logvar_required_for_musplit),
    ] = MicroSplitLossConfig()

    model: Annotated[
        LVAEConfig,
        AfterValidator(lvae_conv_strides_valid),
        AfterValidator(lvae_multiscale_count_valid),
        AfterValidator(lvae_spatial_shape_valid),
        AfterValidator(lvae_depth_valid),
    ]

    noise_model: MultiChannelNMConfig | None = None

    mmse_count: int = Field(default=1, ge=1)

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
        # Remind users to attach a noise model when using denoiSplit
        if self.loss.noise_model_likelihood_weight > 0 and self.noise_model is None:
            warnings.warn(
                "noise_model_likelihood_weight > 0 but no noise_model is provided in "
                "the configuration. A noise model is required for denoiSplit training. "
                "Train one with NoiseModelTrainer, then pass the result to "
                "create_microsplit_config(noise_model=trainer.get_config()) "
                "before training.",
                UserWarning,
                stacklevel=2,
            )

        # Warn about a noise model that will never be used
        if (
            self.noise_model is not None
            and self.loss.noise_model_likelihood_weight == 0
        ):
            warnings.warn(
                "A noise_model is provided but noise_model_likelihood_weight is 0, so "
                "the noise model likelihood is disabled and the noise model will not "
                "be used.",
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

    @classmethod
    def is_supervised(cls) -> bool:
        """
        Return whether the algorithm is supervised.

        Returns
        -------
        bool
            Whether the algorithm is supervised.
        """
        return True
