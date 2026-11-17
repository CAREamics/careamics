"""PN2V Algorithm configuration."""

from typing import Annotated, Literal, Self

from bioimageio.spec.generic.v0_3 import CiteEntry
from pydantic import AfterValidator, ConfigDict, Field, model_validator

from careamics.config.algorithms.n2v_manipulation import N2VManipulateConfig
from careamics.config.architectures import UNetConfig
from careamics.config.support import SupportedPixelManipulation
from careamics.config.validators import (
    model_without_final_activation,
)
from careamics.references.pn2v import (
    N2V2_REF,
    PN2V,
    PN2V2,
    PN2V2_DESCRIPTION,
    PN2V_DESCRIPTION,
    PN2V_REF,
    STRUCT_PN2V,
    STRUCT_PN2V2,
    STRUCT_PN2V2_DESCRIPTION,
    STRUCT_PN2V_DESCRIPTION,
    STRUCTN2V_REF,
)

from .unet_algorithm_config import UNetBasedAlgorithm


def _model_predicts_prior_samples(model: UNetConfig) -> UNetConfig:
    """Validate that the UNet predicts at least 2 prior samples per input channel.

    The outputs are channel-major: input channel `c` owns the outputs
    `[c * K, (c + 1) * K)`, with `K = num_classes // in_channels`.

    Parameters
    ----------
    model : UNetConfig
        Model to validate.

    Returns
    -------
    UNetConfig
        The validated model.

    Raises
    ------
    ValueError
        If `num_classes` is not a multiple of `in_channels`, or if there are fewer
        than 2 prior samples per input channel.
    """
    n_inputs, n_outputs = model.in_channels, model.num_classes
    if n_outputs % n_inputs != 0:
        raise ValueError(
            f"PN2V predicts the same number of prior samples for each input channel, "
            f"so `num_classes` ({n_outputs}) must be a multiple of `in_channels` "
            f"({n_inputs})."
        )
    if n_outputs // n_inputs < 2:
        raise ValueError(
            f"PN2V needs at least 2 prior samples per input channel, but "
            f"`num_classes={n_outputs}` and `in_channels={n_inputs}` give "
            f"{n_outputs // n_inputs}. The PN2V paper uses 800 samples."
        )
    return model


class PN2VAlgorithm(UNetBasedAlgorithm):
    """PN2V Algorithm configuration.

    The UNet predicts `n_prior_samples` prior samples per pixel and input channel, so
    `model.num_classes = model.in_channels * n_prior_samples`. The outputs are
    channel-major: input channel `c` owns the outputs `[c * K, (c + 1) * K)`.
    """

    model_config = ConfigDict(validate_assignment=True)

    algorithm: Literal["pn2v"] = "pn2v"
    """PN2V Algorithm name."""

    loss: Literal["pn2v"] = "pn2v"
    """PN2V loss: negative log of the noise model likelihood of the masked pixels,
    averaged over the prior samples.

    The noise model is not part of the configuration. Pass it to
    `CAREamist.train(noise_model=...)` or `PN2VModule.set_noise_model`.
    """

    monitor_metric: Literal["train_loss", "train_loss_epoch", "val_loss"] = "val_loss"
    """Metric to monitor for the learning rate scheduler."""

    sample_scale: float = Field(default=10.0, gt=0)
    """Scale applied to normalized network outputs before denormalization.

    Matches the upstream PN2V ``outScaling``. The loss and the MMSE estimate both use
    these scaled, denormalized samples.
    """

    n2v_config: N2VManipulateConfig = N2VManipulateConfig()

    model: Annotated[
        UNetConfig,
        # `num_classes` is the number of prior samples times the number of input
        # channels. Do not apply `model_matching_in_out_channels`.
        AfterValidator(model_without_final_activation),
        AfterValidator(_model_predicts_prior_samples),
    ]

    @property
    def n_prior_samples(self) -> int:
        """Number of prior samples predicted per pixel and input channel.

        Returns
        -------
        int
            `model.num_classes // model.in_channels`.
        """
        return self.model.num_classes // self.model.in_channels

    @model_validator(mode="after")
    def validate_n2v2(self) -> Self:
        """Validate that the N2V2 strategy and models are set correctly.

        Returns
        -------
        Self
            The validated configuration.

        Raises
        ------
        ValueError
            If N2V2 is used with the wrong pixel manipulation strategy.
        """
        if self.model.n2v2:
            if self.n2v_config.strategy != SupportedPixelManipulation.MEDIAN.value:
                raise ValueError(
                    f"PN2V2 can only be used with the "
                    f"{SupportedPixelManipulation.MEDIAN} pixel manipulation strategy. "
                    f"Change the `strategy` parameters in `n2v_config` to "
                    f"{SupportedPixelManipulation.MEDIAN}."
                )
        else:
            if self.n2v_config.strategy != SupportedPixelManipulation.UNIFORM.value:
                raise ValueError(
                    f"PN2V can only be used with the "
                    f"{SupportedPixelManipulation.UNIFORM} pixel manipulation strategy."
                    f" Change the `strategy` parameters in `n2v_config` to "
                    f"{SupportedPixelManipulation.UNIFORM}."
                )
        return self

    def set_n2v2(self, use_n2v2: bool) -> None:
        """
        Set the configuration to use PN2V2 or the vanilla Probabilistic Noise2Void.

        This method ensures that PN2V2 is set correctly and remain coherent, as opposed
        to setting the different parameters individually.

        Parameters
        ----------
        use_n2v2 : bool
            Whether to use PN2V2.
        """
        if use_n2v2:
            self.n2v_config.strategy = SupportedPixelManipulation.MEDIAN.value
            self.model.n2v2 = True
        else:
            self.n2v_config.strategy = SupportedPixelManipulation.UNIFORM.value
            self.model.n2v2 = False

    def is_struct_n2v(self) -> bool:
        """Check if the configuration is using structPN2V.

        Returns
        -------
        bool
            Whether the configuration is using structPN2V.
        """
        return self.n2v_config.struct_mask is not None

    def get_algorithm_friendly_name(self) -> str:
        """
        Get the friendly name of the algorithm.

        Returns
        -------
        str
            Friendly name.
        """
        use_n2v2 = self.model.n2v2
        use_structN2V = self.is_struct_n2v()

        if use_n2v2 and use_structN2V:
            return STRUCT_PN2V2
        elif use_n2v2:
            return PN2V2
        elif use_structN2V:
            return STRUCT_PN2V
        else:
            return PN2V

    def get_algorithm_keywords(self) -> list[str]:
        """
        Get algorithm keywords.

        Returns
        -------
        list[str]
            List of keywords.
        """
        use_n2v2 = self.model.n2v2
        use_structN2V = self.is_struct_n2v()

        keywords = [
            "denoising",
            "restoration",
            "UNet",
            "3D" if self.model.is_3D() else "2D",
            "CAREamics",
            "pytorch",
            PN2V,
            "probabilistic",
            "noise model",
        ]

        if use_n2v2:
            keywords.append(PN2V2)
        if use_structN2V:
            keywords.append(STRUCT_PN2V)

        return keywords

    def get_algorithm_references(self) -> str:
        """
        Get the algorithm references.

        This is used to generate the README of the BioImage Model Zoo export.

        Returns
        -------
        str
            Algorithm references.
        """
        use_n2v2 = self.model.n2v2
        use_structN2V = self.is_struct_n2v()

        references = [
            PN2V_REF.text + " doi: " + str(PN2V_REF.doi),
            N2V2_REF.text + " doi: " + str(N2V2_REF.doi),
            STRUCTN2V_REF.text + " doi: " + str(STRUCTN2V_REF.doi),
        ]

        # return the (struct)PN2V(2) references
        if use_n2v2 and use_structN2V:
            return "\n".join(references)
        elif use_n2v2:
            references.pop(-1)
            return "\n".join(references)
        elif use_structN2V:
            references.pop(-2)
            return "\n".join(references)
        else:
            return references[0]

    def get_algorithm_citations(self) -> list[CiteEntry]:
        """
        Return a list of citation entries of the current algorithm.

        This is used to generate the model description for the BioImage Model Zoo.

        Returns
        -------
        List[CiteEntry]
            List of citation entries.
        """
        use_n2v2 = self.model.n2v2
        use_structN2V = self.is_struct_n2v()

        references = [PN2V_REF]

        if use_n2v2:
            references.append(N2V2_REF)

        if use_structN2V:
            references.append(STRUCTN2V_REF)

        return references

    def get_algorithm_description(self) -> str:
        """
        Return a description of the algorithm.

        This method is used to generate the README of the BioImage Model Zoo export.

        Returns
        -------
        str
            Description of the algorithm.
        """
        use_n2v2 = self.model.n2v2
        use_structN2V = self.is_struct_n2v()

        if use_n2v2 and use_structN2V:
            return STRUCT_PN2V2_DESCRIPTION
        elif use_n2v2:
            return PN2V2_DESCRIPTION
        elif use_structN2V:
            return STRUCT_PN2V_DESCRIPTION
        else:
            return PN2V_DESCRIPTION

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
