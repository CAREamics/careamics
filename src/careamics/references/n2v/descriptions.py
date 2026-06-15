"""CAREamics N2V model descriptions."""

N2V = "Noise2Void"
N2V2 = "N2V2"
STRUCT_N2V = "StructN2V"
STRUCT_N2V2 = "StructN2V2"

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
