"""PN2V Algorithm configuration."""

from typing import Annotated, Literal, Self

from bioimageio.spec.generic.v0_3 import CiteEntry
from pydantic import AfterValidator, ConfigDict, model_validator

from careamics.config.architectures import UNetConfig
from careamics.config.noise_model import GaussianMixtureNMConfig
from careamics.config.support import SupportedPixelManipulation, SupportedStructAxis
from careamics.config.transformations import N2VManipulateConfig
from careamics.config.validators import (
    model_without_final_activation,
)

from .unet_algorithm_config import UNetBasedAlgorithm

PN2V = "Probabilistic Noise2Void"
PN2V2 = "PN2V2"
STRUCT_PN2V = "StructPN2V"
STRUCT_PN2V2 = "StructPN2V2"

PN2V_REF = CiteEntry(
    text="Krull, A., Vicar, T., Prakash, M., Lalit, M. and Jug, F., 2020. "
    '"Probabilistic noise2void: Unsupervised content-aware denoising". '
    "Frontiers in Computer Science, 2, p.5.",
    doi="10.3389/fcomp.2020.00005",
)

N2V2_REF = CiteEntry(
    text="Höck, E., Buchholz, T.O., Brachmann, A., Jug, F. and Freytag, A., "
    '2022. "N2V2 - Fixing Noise2Void checkerboard artifacts with modified '
    'sampling strategies and a tweaked network architecture". In European '
    "Conference on Computer Vision (pp. 503-518).",
    doi="10.1007/978-3-031-25069-9_33",
)

STRUCTN2V_REF = CiteEntry(
    text="Broaddus, C., Krull, A., Weigert, M., Schmidt, U. and Myers, G., 2020."
    '"Removing structured noise with self-supervised blind-spot '
    'networks". In 2020 IEEE 17th International Symposium on Biomedical '
    "Imaging (ISBI) (pp. 159-163).",
    doi="10.1109/isbi45749.2020.9098336",
)

PN2V_DESCRIPTION = (
    "Probabilistic Noise2Void (PN2V) is a UNet-based self-supervised algorithm that "
    "extends Noise2Void by incorporating a probabilistic noise model to estimate the "
    "posterior distribution of each pixel more precisely. Like N2V, it uses blind-spot "
    "training where random pixels are selected in patches during training and their "
    "value replaced by a neighboring pixel value. The model is then trained to predict "
    "the original pixel value. PN2V additionally uses a noise model to better capture "
    "the noise characteristics and provide more accurate denoising by modeling the "
    "likelihood of pixel intensities."
)

PN2V2_DESCRIPTION = (
    "PN2V2 is a variant of Probabilistic Noise2Void. "
    + PN2V_DESCRIPTION
    + "\nPN2V2 introduces blur-pool layers and removed skip "
    "connections in the UNet architecture to remove checkboard "
    "artefacts, a common artefacts occurring in Noise2Void variants."
)

STRUCT_PN2V_DESCRIPTION = (
    "StructPN2V is a variant of Probabilistic Noise2Void. "
    + PN2V_DESCRIPTION
    + "\nStructPN2V uses a linear mask (horizontal or vertical) to replace "
    "the pixel values of neighbors of the masked pixels by a random "
    "value. Such masking allows removing 1D structured noise from the "
    "the images, the main failure case of the original N2V variants."
)

STRUCT_PN2V2_DESCRIPTION = (
    "StructPN2V2 is a variant of Probabilistic Noise2Void that uses both "
    "structN2V and N2V2. "
    + PN2V_DESCRIPTION
    + "\nStructPN2V2 uses a linear mask (horizontal or vertical) to replace "
    "the pixel values of neighbors of the masked pixels by a random "
    "value. Such masking allows removing 1D structured noise from the "
    "the images, the main failure case of the original N2V variants."
    "\nPN2V2 introduces blur-pool layers and removed skip connections in "
    "the UNet architecture to remove checkboard artefacts, a common "
    "artefacts occurring in Noise2Void variants."
)


class PN2VAlgorithm(UNetBasedAlgorithm):
    """PN2V Algorithm configuration."""

    model_config = ConfigDict(validate_assignment=True)

    algorithm: Literal["pn2v"] = "pn2v"
    """PN2V Algorithm name."""

    loss: Literal["pn2v"] = "pn2v"
    """PN2V loss function (uses N2V loss with noise model)."""

    n2v_config: N2VManipulateConfig = N2VManipulateConfig()

    noise_model: GaussianMixtureNMConfig
    """Noise model configuration for probabilistic denoising."""

    model: Annotated[
        UNetConfig,
        # AfterValidator(model_matching_in_out_channels),
        # TODO for pn2v channel handling needs to be changed
        AfterValidator(model_without_final_activation),
    ]

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
        return self.n2v_config.struct_mask_axis != SupportedStructAxis.NONE.value

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
            PN2V_REF.text + " doi: " + PN2V_REF.doi,
            N2V2_REF.text + " doi: " + N2V2_REF.doi,
            STRUCTN2V_REF.text + " doi: " + STRUCTN2V_REF.doi,
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
