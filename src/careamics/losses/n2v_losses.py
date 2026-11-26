"""Noise2Void and related losses."""

import torch

from careamics.models.lvae.noise_models import (
    GaussianMixtureNoiseModel,
    MultiChannelNoiseModel,
)


def n2v_loss(
    manipulated_batch: torch.Tensor,
    original_batch: torch.Tensor,
    masks: torch.Tensor,
    *args,
) -> torch.Tensor:
    """
    N2V Loss function described in A Krull et al 2018.

    Parameters
    ----------
    manipulated_batch : torch.Tensor
        Batch after manipulation function applied.
    original_batch : torch.Tensor
        Original images.
    masks : torch.Tensor
        Coordinates of changed pixels.
    *args : Any
        Additional arguments.

    Returns
    -------
    torch.Tensor
        Loss value.
    """
    errors = (original_batch - manipulated_batch) ** 2
    # Average over pixels and batch
    loss = torch.sum(errors * masks) / torch.sum(masks)
    return loss  # TODO change output to dict ?


def pn2v_likelihoods(
    samples: torch.Tensor,
    observations: torch.Tensor,
    noise_model: GaussianMixtureNoiseModel | MultiChannelNoiseModel,
) -> list[torch.Tensor]:
    """Noise model likelihood of the observations under each prior sample.

    The samples are channel-major: with `C` observation channels and `K` samples per
    channel, input channel `c` owns the samples `[c * K, (c + 1) * K)`. Channel `c`
    is evaluated with the noise model of channel `c`.

    Parameters
    ----------
    samples : torch.Tensor
        Prior samples, shape `(B, C * K, ...)`.
    observations : torch.Tensor
        Noisy observations, shape `(B, C, ...)`.
    noise_model : GaussianMixtureNoiseModel or MultiChannelNoiseModel
        Noise model with one channel per observation channel. A single
        `GaussianMixtureNoiseModel` is only valid for one observation channel.

    Returns
    -------
    list of torch.Tensor
        One tensor per channel, each of shape `(B, K, ...)`, with the likelihood
        `p(observation_c | sample_{c, k})`.

    Raises
    ------
    ValueError
        If the number of samples is not a multiple of the number of observation
        channels, or if the noise model does not have one channel per observation
        channel.
    """
    n_channels = observations.shape[1]
    if samples.shape[1] % n_channels != 0:
        raise ValueError(
            f"The number of samples ({samples.shape[1]}) must be a multiple of the "
            f"number of observation channels ({n_channels})."
        )
    n_samples = samples.shape[1] // n_channels

    noise_models: list[GaussianMixtureNoiseModel]
    if isinstance(noise_model, MultiChannelNoiseModel):
        if len(noise_model) != n_channels:
            raise ValueError(
                f"The noise model has {len(noise_model)} channel(s), but the "
                f"observations have {n_channels} channel(s)."
            )
        noise_models = [
            getattr(noise_model, f"nmodel_{channel}") for channel in range(n_channels)
        ]
    else:
        if n_channels != 1:
            raise ValueError(
                f"A single `GaussianMixtureNoiseModel` covers one channel, but the "
                f"observations have {n_channels} channels. Use a "
                f"`MultiChannelNoiseModel`."
            )
        noise_models = [noise_model]

    return [
        channel_model.likelihood(
            observations[:, channel : channel + 1],
            samples[:, channel * n_samples : (channel + 1) * n_samples],
        )
        for channel, channel_model in enumerate(noise_models)
    ]


def pn2v_loss(
    samples: torch.Tensor,
    labels: torch.Tensor,
    masks: torch.Tensor,
    noise_model: GaussianMixtureNoiseModel | MultiChannelNoiseModel,
) -> torch.Tensor:
    """
    Probabilistic N2V loss function described in A Krull et al., CVF (2019).

    For each channel and pixel, the loss is the negative log of the noise model
    likelihood of the original pixel, averaged over the prior samples of that channel.
    The result is averaged over the masked pixels of all channels.

    The samples are channel-major: input channel `c` owns the samples
    `[c * K, (c + 1) * K)`. With one channel, `samples` has shape `(B, K, ...)`.

    Parameters
    ----------
    samples : torch.Tensor
        Prior samples predicted by the network, shape `(B, C * K, ...)`.
    labels : torch.Tensor
        Original pixel values, shape `(B, C, ...)`.
    masks : torch.Tensor
        Coordinates of manipulated pixels, shape `(B, C, ...)`.
    noise_model : GaussianMixtureNoiseModel or MultiChannelNoiseModel
        Noise model with one channel per channel of `labels`. A single
        `GaussianMixtureNoiseModel` is only valid for one channel.

    Returns
    -------
    torch.Tensor
        Loss value.
    """
    likelihoods = pn2v_likelihoods(samples, labels, noise_model)
    likelihoods_avg = torch.cat(
        [
            torch.log(torch.mean(channel_likelihoods, dim=1, keepdim=True))
            for channel_likelihoods in likelihoods
        ],
        dim=1,
    )

    # Average over pixels, channels and batch
    loss = -torch.sum(likelihoods_avg * masks) / torch.sum(masks)
    return loss
