"""HDN algorithm configuration."""

from pprint import pformat
from typing import Annotated, Literal, Self

from bioimageio.spec.generic.v0_3 import CiteEntry
from pydantic import AfterValidator, BaseModel, ConfigDict, model_validator

from careamics.config.architectures import LVAEConfig
from careamics.config.lightning.optimizer_configs import (
    LrSchedulerConfig,
    OptimizerConfig,
)
from careamics.config.losses.loss_config import HDNLossConfig
from careamics.config.noise_model.noise_model_config import MultiChannelNMConfig
from careamics.config.validators import (
    lvae_conv_strides_valid,
    lvae_depth_valid,
    lvae_spatial_shape_valid,
    model_with_single_output_channel,
    model_without_multiscale,
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

    loss: HDNLossConfig = HDNLossConfig()

    model: Annotated[
        LVAEConfig,
        AfterValidator(model_without_multiscale),
        AfterValidator(model_with_single_output_channel),
        AfterValidator(lvae_conv_strides_valid),
        AfterValidator(lvae_spatial_shape_valid),
        AfterValidator(lvae_depth_valid),
    ]

    noise_model: MultiChannelNMConfig | None = None

    # overwrite default optimizer
    optimizer: OptimizerConfig = OptimizerConfig(name="Adamax")
    """Optimizer to use, defined in SupportedOptimizer."""

    lr_scheduler: LrSchedulerConfig = LrSchedulerConfig()

    supervised: bool = False
    """Whether to train against target data instead of the input itself."""

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

    @model_validator(mode="after")
    def validate_supervised_without_noise_model(self: Self) -> Self:
        """Validate that supervised training does not use a noise model.

        Returns
        -------
        Self
            The validated model.

        Raises
        ------
        ValueError
            If `supervised` is `True` and a noise model is provided.
        """
        if self.supervised and self.noise_model is not None:
            raise ValueError(
                "HDN with a noise model is not supported in supervised mode. Remove "
                "the noise model, or set `supervised=False`."
            )
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

    def is_supervised(self) -> bool:
        """
        Return whether the algorithm is supervised.

        Returns
        -------
        bool
            Whether the algorithm is supervised.
        """
        return self.supervised
