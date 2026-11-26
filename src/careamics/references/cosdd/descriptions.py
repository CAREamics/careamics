"""CAREamics COSDD model descriptions."""

COSDD = "COSDD"

COSDD_DESCRIPTION = (
    "COSDD (COrrelated and Signal-Dependent Denoising) is a fully unsupervised "
    "denoiser for noise that is signal-dependent and correlated along rows or "
    "columns of images, as commonly found in scanning-based microscopy and in "
    "EMCCD or sCMOS sensors. A Ladder VAE encodes the clean signal into a latent "
    "code, an autoregressive decoder models the correlated noise and a signal "
    "decoder maps the latent code to image space. Denoised images are obtained by "
    "averaging signal samples or with a direct denoiser trained to predict their "
    "average. It is trained using only noisy images."
)
