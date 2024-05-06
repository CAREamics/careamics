"""Descriptions of the algorithms used in CAREmics."""

from pydantic import BaseModel

CUSTOM = "Custom"
N2V = "Noise2Void"
N2V2 = "N2V2"
STRUCT_N2V = "StructN2V"
STRUCT_N2V2 = "StructN2V2"
N2N = "Noise2Noise"
CARE = "CARE"


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


class AlgorithmDescription(BaseModel):
    """Description of an algorithm.

    Attributes
    ----------
    description : str
        Description of the algorithm.
    """

    description: str


class N2VDescription(AlgorithmDescription):
    """Description of Noise2Void.

    Attributes
    ----------
    description : str
        Description of Noise2Void.
    """

    description: str = N2V_DESCRIPTION


class N2V2Description(AlgorithmDescription):
    """Description of N2V2.

    Attributes
    ----------
    description : str
        Description of N2V2.
    """

    description: str = (
        "N2V2 is a variant of Noise2Void. "
        + N2V_DESCRIPTION
        + "\nN2V2 introduces blur-pool layers and removed skip "
        "connections in the UNet architecture to remove checkboard "
        "artefacts, a common artefacts ocurring in Noise2Void."
    )


class StructN2VDescription(AlgorithmDescription):
    """Description of StructN2V.

    Attributes
    ----------
    description : str
        Description of StructN2V.
    """

    description: str = (
        "StructN2V is a variant of Noise2Void. "
        + N2V_DESCRIPTION
        + "\nStructN2V uses a linear mask (horizontal or vertical) to replace "
        "the pixel values of neighbors of the masked pixels by a random "
        "value. Such masking allows removing 1D structured noise from the "
        "the images, the main failure case of the original N2V."
    )


class StructN2V2Description(AlgorithmDescription):
    """Description of StructN2V2.

    Attributes
    ----------
    description : str
        Description of StructN2V2.
    """

    description: str = (
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


class N2NDescription(AlgorithmDescription):
    """Description of Noise2Noise.

    Attributes
    ----------
    description : str
        Description of Noise2Noise.
    """

    description: str = "Noise2Noise"  # TODO


class CAREDescription(AlgorithmDescription):
    """Description of CARE.

    Attributes
    ----------
    description : str
        Description of CARE.
    """

    description: str = "CARE"  # TODO
