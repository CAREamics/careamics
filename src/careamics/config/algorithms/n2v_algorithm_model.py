"""N2V Algorithm configuration."""

from typing import Annotated, Literal, Self

from bioimageio.spec.generic.v0_3 import CiteEntry
from pydantic import AfterValidator, ConfigDict, model_validator

from careamics.config.architectures import UNetModel
from careamics.config.support import SupportedPixelManipulation, SupportedStructAxis
from careamics.config.transformations import N2VManipulateModel
from careamics.config.validators import (
    model_matching_in_out_channels,
    model_without_final_activation,
)

from .unet_algorithm_model import UNetBasedAlgorithm

N2V = "Noise2Void"
N2V2 = "N2V2"
STRUCT_N2V = "StructN2V"
STRUCT_N2V2 = "StructN2V2"

N2V_REF = CiteEntry(
    text='Krull, A., Buchholz, T.O. and Jug, F., 2019. "Noise2Void - Learning '
    'denoising from single noisy images". In Proceedings of the IEEE/CVF '
    "conference on computer vision and pattern recognition (pp. 2129-2137).",
    doi="10.1109/cvpr.2019.00223",
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

N2V_DESCRIPTION = (
    "Noise2Void is a UNet-based self-supervised algorithm that "
    "uses blind-spot training to denoise images. In short, in every "
    "patches during training, random pixels are selected and their "
    "value replaced by a neighboring pixel value. The network is then "
    "trained to predict the original pixel value. The algorithm "
    "relies on the continuity of the signal (neighboring pixels have "
    "similar values) and the pixel-wise independence of the noise "
    "(the noise in a pixel is not correlated with the noise in "
    "neighboring pixels)."
)

N2V2_DESCRIPTION = (
    "N2V2 is a variant of Noise2Void. "
    + N2V_DESCRIPTION
    + "\nN2V2 introduces blur-pool layers and removed skip "
    "connections in the UNet architecture to remove checkboard "
    "artefacts, a common artefacts ocurring in Noise2Void."
)

STR_N2V_DESCRIPTION = (
    "StructN2V is a variant of Noise2Void. "
    + N2V_DESCRIPTION
    + "\nStructN2V uses a linear mask (horizontal or vertical) to replace "
    "the pixel values of neighbors of the masked pixels by a random "
    "value. Such masking allows removing 1D structured noise from the "
    "the images, the main failure case of the original N2V."
)

STR_N2V2_DESCRIPTION = (
    "StructN2V2 is a a variant of Noise2Void that uses both "
    "structN2V and N2V2. "
    + N2V_DESCRIPTION
    + "\nStructN2V2 uses a linear mask (horizontal or vertical) to replace "
    "the pixel values of neighbors of the masked pixels by a random "
    "value. Such masking allows removing 1D structured noise from the "
    "the images, the main failure case of the original N2V."
    "\nN2V2 introduces blur-pool layers and removed skip connections in "
    "the UNet architecture to remove checkboard artefacts, a common "
    "artefacts ocurring in Noise2Void."
)


class N2VAlgorithm(UNetBasedAlgorithm):
    """N2V Algorithm configuration."""

    model_config = ConfigDict(validate_assignment=True)

    algorithm: Literal["n2v"] = "n2v"
    """N2V Algorithm name."""

    loss: Literal["n2v"] = "n2v"
    """N2V loss function."""

    n2v_config: N2VManipulateModel = N2VManipulateModel()

    model: Annotated[
        UNetModel,
        AfterValidator(model_matching_in_out_channels),
        AfterValidator(model_without_final_activation),
    ]

    @model_validator(mode="after")
    def validate_n2v2(self) -> Self:
        """Validate that the N2V2 strategy and models are set correctly.

        Returns
        -------
        Self
            The validateed configuration.

        Raises
        ------
        ValueError
            If N2V2 is used with the wrong pixel manipulation strategy.
        """
        if self.model.n2v2:
            if self.n2v_config.strategy != SupportedPixelManipulation.MEDIAN.value:
                raise ValueError(
                    f"N2V2 can only be used with the "
                    f"{SupportedPixelManipulation.MEDIAN} pixel manipulation strategy. "
                    f"Change the `strategy` parameters in `n2v_config` to "
                    f"{SupportedPixelManipulation.MEDIAN}."
                )
        else:
            if self.n2v_config.strategy != SupportedPixelManipulation.UNIFORM.value:
                raise ValueError(
                    f"N2V can only be used with the "
                    f"{SupportedPixelManipulation.UNIFORM} pixel manipulation strategy."
                    f" Change the `strategy` parameters in `n2v_config` to "
                    f"{SupportedPixelManipulation.UNIFORM}."
                )
        return self

    def set_n2v2(self, use_n2v2: bool) -> None:
        """
        Set the configuration to use N2V2 or the vanilla Noise2Void.

        This method ensures that N2V2 is set correctly and remain coherent, as opposed
        to setting the different parameters individually.

        Parameters
        ----------
        use_n2v2 : bool
            Whether to use N2V2.
        """
        if use_n2v2:
            self.n2v_config.strategy = SupportedPixelManipulation.MEDIAN.value
            self.model.n2v2 = True
        else:
            self.n2v_config.strategy = SupportedPixelManipulation.UNIFORM.value
            self.model.n2v2 = False

    def is_struct_n2v(self) -> bool:
        """Check if the configuration is using structN2V.

        Returns
        -------
        bool
            Whether the configuration is using structN2V.
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
            return STRUCT_N2V2
        elif use_n2v2:
            return N2V2
        elif use_structN2V:
            return STRUCT_N2V
        else:
            return N2V

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
            N2V,
        ]

        if use_n2v2:
            keywords.append(N2V2)
        if use_structN2V:
            keywords.append(STRUCT_N2V)

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
            N2V_REF.text + " doi: " + N2V_REF.doi,
            N2V2_REF.text + " doi: " + N2V2_REF.doi,
            STRUCTN2V_REF.text + " doi: " + STRUCTN2V_REF.doi,
        ]

        # return the (struct)N2V(2) references
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

        references = [N2V_REF]

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
            return STR_N2V2_DESCRIPTION
        elif use_n2v2:
            return N2V2_DESCRIPTION
        elif use_structN2V:
            return STR_N2V_DESCRIPTION
        else:
            return N2V_DESCRIPTION
