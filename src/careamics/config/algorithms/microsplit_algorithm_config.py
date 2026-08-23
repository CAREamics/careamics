"""MicroSplit algorithm configuration."""

from pprint import pformat
from typing import Annotated, Literal, Self

from bioimageio.spec.generic.v0_3 import CiteEntry
from pydantic import AfterValidator, BaseModel, ConfigDict, model_validator

from careamics.config.architectures import LVAEConfig
from careamics.config.lightning.optimizer_configs import (
    LrSchedulerConfig,
    OptimizerConfig,
)
from careamics.config.losses.loss_config import MicroSplitLossConfig
from careamics.config.validators import (
    lvae_conv_strides_valid,
    lvae_depth_valid,
    lvae_multiscale_count_valid,
    lvae_spatial_shape_valid,
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

    loss: MicroSplitLossConfig = MicroSplitLossConfig()

    model: Annotated[
        LVAEConfig,
        AfterValidator(lvae_conv_strides_valid),
        AfterValidator(lvae_multiscale_count_valid),
        AfterValidator(lvae_spatial_shape_valid),
        AfterValidator(lvae_depth_valid),
    ]

    optimizer: OptimizerConfig = OptimizerConfig()
    """Optimizer to use, defined in SupportedOptimizer."""

    lr_scheduler: LrSchedulerConfig = LrSchedulerConfig()

    @model_validator(mode="after")
    def validate_constraints(self: Self) -> Self:
        """Validate the algorithm-specific constraints.

        The noise model is not part of the configuration; it is supplied at training
        time (`CAREamist.train(noise_model=...)` / `MicroSplitModule.set_noise_model`)
        and validated against the model then. Only config-only constraints are checked
        here.

        Returns
        -------
        Self
            The validated model.

        Raises
        ------
        ValueError
            If the loss and model configurations are not compatible with the MicroSplit
            algorithm.
        """
        predict_logvar_consistent(self.model, self.loss)
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
