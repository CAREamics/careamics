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
from careamics.config.losses.loss_config import LVAELossConfig
from careamics.config.noise_model.noise_model_config import MultiChannelNMConfig
from careamics.config.validators import (
    loss_type_is_microsplit,
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

    loss: Annotated[LVAELossConfig, AfterValidator(loss_type_is_microsplit)] = (
        LVAELossConfig(loss_type="microsplit")
    )

    model: LVAEConfig

    noise_model: MultiChannelNMConfig | None = None

    mmse_count: int = Field(default=1, ge=1)

    optimizer: OptimizerConfig = OptimizerConfig()
    """Optimizer to use, defined in SupportedOptimizer."""

    lr_scheduler: LrSchedulerConfig = LrSchedulerConfig()

    @model_validator(mode="after")
    def warn_denoisplit_without_noise_model(self: Self) -> Self:
        """Remind users to attach a noise model when using denoiSplit.

        Returns
        -------
        Self
            The validated model.

        Warns
        -----
        UserWarning
            If `denoisplit_weight` is greater than 0 and no noise model is provided.
        """
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
        return self

    @model_validator(mode="after")
    def validate_predict_logvar(self: Self) -> Self:
        """Validate the consistency of `predict_logvar` between model and loss.

        Returns
        -------
        Self
            The validated model.

        Raises
        ------
        ValueError
            If the model and loss `predict_logvar` do not match.
        """
        predict_logvar_consistent(self.model, self.loss)
        return self

    @model_validator(mode="after")
    def validate_noise_model_channels(self: Self) -> Self:
        """Validate that the number of noise models matches the output channels.

        Returns
        -------
        Self
            The validated model.

        Raises
        ------
        ValueError
            If the number of output channels does not match the number of noise
            models.
        """
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
