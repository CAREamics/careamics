"""CryoCARE algorithm configuration."""

from typing import Annotated, Literal

from bioimageio.spec.generic.v0_3 import CiteEntry
from pydantic import AfterValidator, Field

from careamics.config.architectures import UNetConfig
from careamics.config.lightning.optimizer_configs import OptimizerConfig
from careamics.references.cryocare import (
    CRYOCARE,
    CRYOCARE_DESCRIPTION,
    CRYOCARE_REF,
)

from .unet_algorithm_config import UNetBasedAlgorithm


def _default_cryocare_model() -> UNetConfig:
    """Create the default CryoCARE U-Net configuration.

    Returns
    -------
    UNetConfig
        Default three-dimensional U-Net configuration for CryoCARE.
    """
    return UNetConfig(
        architecture="UNet",
        conv_dims=3,
        in_channels=1,
        num_classes=1,
        depth=3,
        num_channels_init=16,
        residual=True,
        final_activation="None",
        n2v2=False,
        independent_channels=False,
        use_batch_norm=False,
    )


def _model_with_cryocare_constraints(model: UNetConfig) -> UNetConfig:
    """Validate the U-Net constraints required by CryoCARE.

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
        If the model is incompatible with the cryoCARE architecture.
    """
    expected = {
        "conv_dims": 3,
        "in_channels": 1,
        "num_classes": 1,
        "depth": 3,
        "num_channels_init": 16,
        "residual": True,
        "final_activation": "None",
        "n2v2": False,
        "independent_channels": False,
        "use_batch_norm": False,
    }
    invalid = {
        name: (getattr(model, name), value)
        for name, value in expected.items()
        if getattr(model, name) != value
    }
    if invalid:
        details = ", ".join(
            f"{name}={actual!r} (expected {required!r})"
            for name, (actual, required) in invalid.items()
        )
        raise ValueError(f"Invalid cryoCARE UNet configuration: {details}.")

    return model


class CryoCAREAlgorithm(UNetBasedAlgorithm):
    """CryoCARE Noise2Noise algorithm configuration."""

    algorithm: Literal["cryocare"] = "cryocare"
    """CryoCARE algorithm name."""

    loss: Literal["mse"] = "mse"
    """Loss used by CryoCARE."""

    model: Annotated[UNetConfig, AfterValidator(_model_with_cryocare_constraints)] = (
        Field(default_factory=_default_cryocare_model)
    )
    """The constrained three-dimensional U-Net used by CryoCARE."""

    optimizer: OptimizerConfig = Field(
        default_factory=lambda: OptimizerConfig(name="Adam", parameters={"lr": 4e-4})
    )
    """Adam optimizer with CryoCARE's reference learning rate."""

    def get_algorithm_friendly_name(self) -> str:
        """Return the friendly algorithm name.

        Returns
        -------
        str
            Friendly name of the algorithm.
        """
        return CRYOCARE

    def get_algorithm_keywords(self) -> list[str]:
        """Return algorithm keywords.

        Returns
        -------
        list[str]
            List of algorithm keywords.
        """
        return [
            "restoration",
            "tomogram",
            "UNet",
            "3D",
            "CAREamics",
            "pytorch",
            CRYOCARE,
        ]

    def get_algorithm_references(self) -> str:
        """Return the algorithm reference.

        Returns
        -------
        str
            Algorithm reference and DOI.
        """
        return CRYOCARE_REF.text + " doi: " + str(CRYOCARE_REF.doi)

    def get_algorithm_citations(self) -> list[CiteEntry]:
        """Return citation entries for CryoCARE.

        Returns
        -------
        list[CiteEntry]
            Citation entries for the algorithm.
        """
        return [CRYOCARE_REF]

    def get_algorithm_description(self) -> str:
        """Return the algorithm description.

        Returns
        -------
        str
            Description of the algorithm.
        """
        return CRYOCARE_DESCRIPTION

    @classmethod
    def is_supervised(cls) -> bool:
        """Return whether CryoCARE uses paired targets during training.

        Returns
        -------
        bool
            Whether the algorithm is supervised.
        """
        return True
