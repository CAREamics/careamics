"""CAREamics PN2V model descriptions."""

PN2V = "Probabilistic Noise2Void"
PN2V2 = "PN2V2"
STRUCT_PN2V = "StructPN2V"
STRUCT_PN2V2 = "StructPN2V2"

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
