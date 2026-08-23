"""Noise models for LVAE-based algorithms."""

from __future__ import annotations

import copy
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray

from careamics.utils import get_device

if TYPE_CHECKING:
    from careamics.config import GaussianMixtureNMConfig, MultiChannelNMConfig

# TODO this module shouldn't be in lvae folder


def create_histogram(
    bins: int, min_val: float, max_val: float, observation: NDArray, signal: NDArray
) -> NDArray:
    """
    Create a 2D histogram from 'observation' and 'signal'.

    Parameters
    ----------
    bins : int
        Number of bins in x and y.
    min_val : float
        Lower bound of the lowest bin in x and y.
    max_val : float
        Upper bound of the highest bin in x and y.
    observation : np.ndarray
        3D numpy array (stack of 2D images).
        Observation.shape[0] must be divisible by signal.shape[0].
        Assumes that n subsequent images in observation belong to one image in 'signal'.
    signal : np.ndarray
        3D numpy array (stack of 2D images).

    Returns
    -------
    np.ndarray
        A 3D array:
        - histogram[0]: Normalized 2D counts.
        - histogram[1]: Lower boundaries of bins along y.
        - histogram[2]: Upper boundaries of bins along y.
        The values for x can be obtained by transposing 'histogram[1]' and
        'histogram[2]'.
    """
    histogram = np.zeros((3, bins, bins))

    value_range = [min_val, max_val]

    # Compute mapping factor between observation and signal samples
    obs_to_signal_shape_factor = int(observation.shape[0] / signal.shape[0])

    # Flatten arrays and align signal values
    signal_indices = np.arange(observation.shape[0]) // obs_to_signal_shape_factor
    signal_values = signal[signal_indices].ravel()
    observation_values = observation.ravel()

    count_histogram, signal_edges, _ = np.histogram2d(
        signal_values, observation_values, bins=bins, range=[value_range, value_range]
    )

    # Normalize rows to obtain probabilities
    row_sums = count_histogram.sum(axis=1, keepdims=True)
    count_histogram /= np.clip(row_sums, a_min=1e-20, a_max=None)

    histogram[0] = count_histogram
    histogram[1] = signal_edges[:-1][..., np.newaxis]
    histogram[2] = signal_edges[1:][..., np.newaxis]

    return histogram


def noise_model_factory(
    model_config: GaussianMixtureNMConfig | None,
) -> GaussianMixtureNoiseModel | None:
    """Noise model factory for single-channel noise models.

    Parameters
    ----------
    model_config : GaussianMixtureNMConfig | None
        Noise model configuration for a single Gaussian mixture noise model.

    Returns
    -------
    GaussianMixtureNoiseModel | None
        A single noise model instance, or None if no config is provided.

    Raises
    ------
    NotImplementedError
        If the chosen noise model `model_type` is not implemented.
        Currently only `GaussianMixtureNoiseModel` is implemented.
    """
    if model_config:
        if model_config.model_type == "GaussianMixtureNoiseModel":
            return GaussianMixtureNoiseModel(model_config)
        else:
            raise NotImplementedError(
                f"Model {model_config.model_type} is not implemented"
            )
    return None


def multichannel_noise_model_factory(
    model_config: MultiChannelNMConfig | None,
) -> MultiChannelNoiseModel | None:
    """Multi-channel noise model factory.

    Parameters
    ----------
    model_config : MultiChannelNMConfig | None
        Noise model configuration, a `MultiChannelNMConfig` config that defines
        noise models for the different output channels.

    Returns
    -------
    MultiChannelNoiseModel | None
        A noise model instance, or None if no config is provided.

    Raises
    ------
    NotImplementedError
        If the chosen noise model `model_type` is not implemented.
        Currently only `GaussianMixtureNoiseModel` is implemented.
    """
    if model_config:
        noise_models = []
        for nm in model_config.noise_models:
            if nm.model_type == "GaussianMixtureNoiseModel":
                noise_models.append(GaussianMixtureNoiseModel(nm))
            else:
                raise NotImplementedError(f"Model {nm.model_type} is not implemented")
        return MultiChannelNoiseModel(noise_models)
    return None


def train_gm_noise_model(
    model_config: GaussianMixtureNMConfig,
    signal: np.ndarray,
    observation: np.ndarray,
) -> GaussianMixtureNoiseModel:
    """Train a Gaussian mixture noise model.

    Parameters
    ----------
    model_config : GaussianMixtureNMConfig
        Configuration of the Gaussian mixture noise model to train.
    signal : np.ndarray
        Clean signal data.
    observation : np.ndarray
        Noisy observation data.

    Returns
    -------
    GaussianMixtureNoiseModel
        The trained noise model.
    """
    # TODO where to put train params?
    # TODO any training params ? Different channels ?
    noise_model = GaussianMixtureNoiseModel(model_config)
    # TODO revisit config unpacking
    noise_model.fit(signal, observation)
    return noise_model


class MultiChannelNoiseModel(nn.Module):
    """Noise model that wraps one noise model per output channel.

    To handle noise models and the relative likelihood computation for multiple
    output channels (e.g., muSplit, denoiseSplit).

    This class:
    - receives as input a variable number of noise models, one for each channel.
    - computes the likelihood of observations given signals for each channel.
    - returns the concatenation of these likelihoods.

    Parameters
    ----------
    nmodels : list[GaussianMixtureNoiseModel]
        List of noise models, one for each output channel.
    """

    def __init__(self, nmodels: list[GaussianMixtureNoiseModel]):
        """Constructor.

        Parameters
        ----------
        nmodels : list[GaussianMixtureNoiseModel]
            List of noise models, one for each output channel.
        """
        super().__init__()
        self.device = get_device()

        for i, nmodel in enumerate(nmodels):  # TODO refactor this !!!
            if nmodel is not None:
                self.add_module(
                    f"nmodel_{i}", nmodel
                )  # TODO: wouldn't be easier to use a list?

        self._nm_cnt = 0
        for nmodel in nmodels:
            if nmodel is not None:
                self._nm_cnt += 1

        print(f"[{self.__class__.__name__}] Nmodels count:{self._nm_cnt}")

    def __len__(self) -> int:
        """Return the number of per-channel noise models.

        Returns
        -------
        int
            The number of channels the noise model covers.
        """
        return self._nm_cnt

    @classmethod
    def from_npz(cls, paths: Sequence[str | Path]) -> MultiChannelNoiseModel:
        """Build a multi-channel noise model from per-channel ``.npz`` files.

        Each file is a raw-space Gaussian-mixture noise model saved by
        ``GaussianMixtureNoiseModel.save`` / ``NoiseModelTrainer.save``. The files are
        loaded in the given order (one per output channel); when the files carry a
        ``channel_index``, ordering consistency is validated by
        ``MultiChannelNMConfig``.

        Parameters
        ----------
        paths : Sequence[str or Path]
            Paths to the per-channel ``.npz`` noise-model files.

        Returns
        -------
        MultiChannelNoiseModel
            The assembled multi-channel noise model.

        Raises
        ------
        ValueError
            If ``paths`` is empty.
        """
        from careamics.config.noise_model.noise_model_config import (
            GaussianMixtureNMConfig,
            MultiChannelNMConfig,
        )

        if len(paths) == 0:
            raise ValueError("No noise model paths provided.")
        configs = [GaussianMixtureNMConfig.from_npz(Path(p)) for p in paths]
        # MultiChannelNMConfig validates channel-index ordering / count consistency
        config = MultiChannelNMConfig(noise_models=configs)
        model = multichannel_noise_model_factory(config)
        assert model is not None
        return model

    def to_device(self, device: torch.device) -> None:
        """Move this model and all per-channel noise models to `device`.

        Parameters
        ----------
        device : torch.device
            Device to move the model to.
        """
        self.device = device
        self.to(device)
        for ch_idx in range(self._nm_cnt):
            nmodel = getattr(self, f"nmodel_{ch_idx}")
            nmodel.to_device(device)

    def likelihood(self, obs: torch.Tensor, signal: torch.Tensor) -> torch.Tensor:
        """Compute the likelihood of observations given signals for each channel.

        Parameters
        ----------
        obs : torch.Tensor
            Noisy observations, i.e., the target(s). Specifically, the input noisy
            image for HDN, or the noisy unmixed images used for supervision
            for denoiSplit. Shape: (B, C, [Z], Y, X), where C is the number of
            unmixed channels.
        signal : torch.Tensor
            Underlying signals, i.e., the (clean) output of the model. Specifically, the
            denoised image for HDN, or the unmixed images for denoiSplit.
            Shape: (B, C, [Z], Y, X), where C is the number of unmixed channels.

        Returns
        -------
        torch.Tensor
            Concatenation of the per-channel likelihoods, with the same shape
            as the inputs.
        """
        # Case 1: obs and signal have a single channel (e.g., denoising)
        if obs.shape[1] == 1:
            assert signal.shape[1] == 1
            return self.nmodel_0.likelihood(obs, signal)

        # Case 2: obs and signal have multiple channels (e.g., denoiSplit)
        assert obs.shape[1] == self._nm_cnt, (
            "The number of channels in `obs` must match the number of noise models."
            f" Got instead: obs={obs.shape[1]},  nm={self._nm_cnt}"
        )
        ll_list = []
        for ch_idx in range(obs.shape[1]):
            nmodel = getattr(self, f"nmodel_{ch_idx}")
            ll_list.append(
                nmodel.likelihood(
                    obs[:, ch_idx : ch_idx + 1], signal[:, ch_idx : ch_idx + 1]
                )  # slicing to keep the channel dimension
            )
        return torch.cat(ll_list, dim=1)

    def sample_observation(self, signal: NDArray) -> NDArray:
        """Sample noisy observations from the learned noise models.

        For each channel, samples noisy observations using the corresponding
        channel's noise model.

        Parameters
        ----------
        signal : NDArray
            Clean signal data with shape (..., C, Y, X) where C is the number
            of channels matching the number of noise models.

        Returns
        -------
        NDArray
            Sampled noisy observation with same shape as input signal.
        """
        if signal.ndim < 3:
            raise ValueError(
                f"Signal must have at least 3 dimensions (C, Y, X), got {signal.ndim}D"
            )

        n_channels = signal.shape[-3]
        if n_channels != self._nm_cnt:
            raise ValueError(
                f"Number of channels ({n_channels}) must match number of "
                f"noise models ({self._nm_cnt})"
            )

        samples_list = []
        for ch_idx in range(n_channels):
            nmodel = getattr(self, f"nmodel_{ch_idx}")
            channel_signal = signal[..., ch_idx, :, :]
            channel_sample = nmodel.sample_observation_from_signal(channel_signal)
            samples_list.append(channel_sample)

        return np.stack(samples_list, axis=-3)

    @property
    def is_normalized(self) -> bool:
        """Whether all wrapped per-channel noise models are normalized.

        Returns
        -------
        bool
            True if every child model operates in normalized data space.
        """
        return all(
            getattr(self, f"nmodel_{ch_idx}").is_normalized
            for ch_idx in range(self._nm_cnt)
        )

    def get_normalized_copy(
        self, data_means: Sequence[float], data_stds: Sequence[float]
    ) -> MultiChannelNoiseModel:
        """Return a copy with each channel's model in normalized data space.

        Each per-channel noise model is transformed with that channel's
        statistics (see `GaussianMixtureNoiseModel.get_normalized_copy`).
        Length-1 statistics are broadcast to all channels.

        Parameters
        ----------
        data_means : Sequence[float]
            Per-channel means used to normalize the data. Length must be 1 or
            match the number of noise models.
        data_stds : Sequence[float]
            Per-channel standard deviations used to normalize the data. Length
            must be 1 or match the number of noise models.

        Returns
        -------
        MultiChannelNoiseModel
            A new multi-channel model operating on normalized values.

        Raises
        ------
        ValueError
            If the statistics lengths do not match the number of noise models.
        """
        means = list(data_means)
        stds = list(data_stds)
        if len(means) == 1:
            means = means * self._nm_cnt
        if len(stds) == 1:
            stds = stds * self._nm_cnt
        if len(means) != self._nm_cnt or len(stds) != self._nm_cnt:
            raise ValueError(
                f"Number of data means ({len(means)}) and stds ({len(stds)}) "
                f"must be 1 or match the number of noise models "
                f"({self._nm_cnt})."
            )
        return MultiChannelNoiseModel(
            [
                getattr(self, f"nmodel_{ch_idx}").get_normalized_copy(
                    means[ch_idx], stds[ch_idx]
                )
                for ch_idx in range(self._nm_cnt)
            ]
        )


class GaussianMixtureNoiseModel(nn.Module):
    """Define a noise model parameterized as a mixture of gaussians.

    If `config.weight` is provided, the model is initialized from those weights.
    Otherwise weights are randomly initialized using `config.min_signal` and
    `config.max_signal`.

    Parameters
    ----------
    config : GaussianMixtureNMConfig
        A `pydantic` model that defines the configuration of the GMM noise model.

    Attributes
    ----------
    min_signal : float
        Minimum signal intensity expected in the image.
    max_signal : float
        Maximum signal intensity expected in the image.
    weight : torch.nn.Parameter
        A [3*n_gaussian, n_coeff] sized array containing the values of the weights
        describing the GMM noise model, with each row corresponding to one
        parameter of each gaussian, namely [mean, standard deviation and weight].
        Specifically, rows are organized as follows:
        - first n_gaussian rows correspond to the means
        - next n_gaussian rows correspond to the weights
        - last n_gaussian rows correspond to the standard deviations
        If `weight=None`, the weight array is initialized using the `min_signal`
        and `max_signal` parameters.
    n_gaussian: int
        Number of gaussians in the mixture.
    n_coeff: int
        Number of coefficients to describe the functional relationship between gaussian
        parameters and the signal. 2 implies a linear, 3 implies a quadratic
        relationship and so on.
    device: device
        GPU device.
    min_sigma: float
        All values of `standard deviation` below this are clamped to this value.
    """

    # buffers registered in __init__, annotated for mypy
    min_signal: torch.Tensor
    max_signal: torch.Tensor
    min_sigma: torch.Tensor
    tolerance: torch.Tensor

    # TODO training a NM relies on getting a clean data(N2V e.g,)
    def __init__(self, config: GaussianMixtureNMConfig) -> None:
        """Constructor.

        Parameters
        ----------
        config : GaussianMixtureNMConfig
            A `pydantic` model that defines the configuration of the GMM noise model.
        """
        super().__init__()
        self.device = torch.device("cpu")

        params = config.model_dump(exclude_none=True)

        min_sigma = torch.tensor(params["min_sigma"])
        min_signal = torch.tensor(params["min_signal"])
        max_signal = torch.tensor(params["max_signal"])
        self.register_buffer("min_signal", min_signal)
        self.register_buffer("max_signal", max_signal)
        self.register_buffer("min_sigma", min_sigma)
        self.register_buffer("tolerance", torch.tensor([1e-10]))

        # Use config.weight directly to avoid Array PlainSerializer converting
        # the numpy array to a JSON string via model_dump().
        if config.weight is not None:
            weight = torch.as_tensor(np.asarray(config.weight), dtype=torch.float32)
        else:
            weight = self._initialize_weights(
                params["n_gaussian"], params["n_coeff"], max_signal, min_signal
            )

        self.n_gaussian = weight.shape[0] // 3
        self.n_coeff = weight.shape[1]

        self.register_parameter("weight", nn.Parameter(weight))
        self._set_model_mode(mode="prediction")

        # Normalization state
        self.is_normalized: bool = False
        self.normalization_mean: float | None = None
        self.normalization_std: float | None = None

        print(f"[{self.__class__.__name__}] min_sigma: {self.min_sigma}")

    def get_normalized_copy(
        self, data_mean: float, data_std: float
    ) -> GaussianMixtureNoiseModel:
        """Return a copy of this model transformed into normalized data space.

        The GMM parameterization is closed under the joint affine transform
        ``x -> (x - data_mean) / data_std`` of signal and observation, so the
        returned model computes, exactly, ``likelihood_norm(o', s') = data_std *
        likelihood_raw(o, s)`` for normalized inputs. Gradients with respect to
        the normalized signal are identical to denormalizing and evaluating this
        (raw-space) model.

        Parameters
        ----------
        data_mean : float
            Mean used to normalize the data.
        data_std : float
            Standard deviation used to normalize the data. Must be positive.

        Returns
        -------
        GaussianMixtureNoiseModel
            A new model operating on normalized signal/observation values. This
            model is unchanged.

        Raises
        ------
        ValueError
            If `data_std` is not positive or this model is already normalized.
        """
        if data_std <= 0:
            raise ValueError(f"data_std must be positive, got {data_std}.")
        if self.is_normalized:
            raise ValueError(
                "This noise model is already normalized; refusing to normalize "
                "twice. Build a raw model from the configuration instead."
            )

        new = copy.deepcopy(self)
        d = float(data_std)
        m = float(data_mean)
        k = self.n_gaussian
        with torch.no_grad():
            # mean-polynomial rows: the residual (poly - alpha-weighted mean of
            # polys) scales by 1/d; the `+ signal` term follows the normalized
            # signal automatically
            new.weight.data[:k, :] /= d
            # rows K:2K hold w with exp(w) the VARIANCE-polynomial coefficients
            new.weight.data[k : 2 * k, :] -= math.log(d**2)
            # alpha rows are functions of the affine-invariant normalized signal
            # coordinate and are normalized to sum to 1: unchanged
            new.min_signal.copy_((new.min_signal - m) / d)
            new.max_signal.copy_((new.max_signal - m) / d)
            # min_sigma clamps the VARIANCE (despite its name)
            new.min_sigma.copy_(new.min_sigma / d**2)

        new.is_normalized = True
        new.normalization_mean = m
        new.normalization_std = d
        return new

    def _initialize_weights(
        self,
        n_gaussian: int,
        n_coeff: int,
        max_signal: torch.Tensor,
        min_signal: torch.Tensor,
    ) -> torch.Tensor:
        """Create random weight initialization.

        Parameters
        ----------
        n_gaussian : int
            Number of gaussians in the mixture.
        n_coeff : int
            Number of polynomial coefficients per gaussian parameter.
        max_signal : torch.Tensor
            Maximum signal intensity expected in the image.
        min_signal : torch.Tensor
            Minimum signal intensity expected in the image.

        Returns
        -------
        torch.Tensor
            Randomly initialized weights of shape [3 * n_gaussian, n_coeff].
        """
        weight = torch.randn(n_gaussian * 3, n_coeff)
        weight[n_gaussian : 2 * n_gaussian, 1] = torch.log(
            max_signal - min_signal
        ).float()
        return weight

    def to_device(self, device: torch.device) -> None:
        """Move the model to `device`.

        Parameters
        ----------
        device : torch.device
            Device to move the model to.
        """
        self.device = device
        self.to(device)

    def _set_model_mode(self, mode: str) -> None:
        """Set `requires_grad` on the weights depending on the mode.

        Parameters
        ----------
        mode : str
            Either "train" or "prediction".
        """
        if mode == "train":
            self.weight.requires_grad = True
        else:
            self.weight.requires_grad = False

    def polynomial_regressor(
        self, weight_params: torch.Tensor, signals: torch.Tensor
    ) -> torch.Tensor:
        """Combine `weight_params` and `signals` to regress for the gaussian params.

        Parameters
        ----------
        weight_params : torch.Tensor
            Corresponds to specific rows of `self.weight`.
        signals : torch.Tensor
            Signals.

        Returns
        -------
        torch.Tensor
            Corresponds to either of mean, std or weight, evaluated at `signals`.
        """
        value = torch.zeros_like(signals)
        device = (
            value.device
        )  # TODO the whole device handling in this class needs to be refactored
        weight_params = weight_params.to(device)
        self.min_signal = self.min_signal.to(device)
        self.max_signal = self.max_signal.to(device)
        for i in range(weight_params.shape[0]):
            value += weight_params[i] * (
                ((signals - self.min_signal) / (self.max_signal - self.min_signal)) ** i
            )
        return value

    def normal_density(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        """
        Evaluate the normal probability density at `x` given the `mean` and `std`.

        Parameters
        ----------
        x : torch.Tensor
            The ground-truth tensor. Shape is (batch, 1, dim1, dim2).
        mean : torch.Tensor
            The inferred mean of distribution. Shape is (batch, 1, dim1, dim2).
        std : torch.Tensor
            The inferred std of distribution. Shape is (batch, 1, dim1, dim2).

        Returns
        -------
        torch.Tensor
            Normal probability density of `x` given `mean` and `std`.
        """
        tmp = -((x - mean) ** 2)
        tmp = tmp / (2.0 * std * std)
        tmp = torch.exp(tmp)
        tmp = tmp / torch.sqrt((2.0 * np.pi) * std * std)
        return tmp

    def likelihood(
        self, observations: torch.Tensor, signals: torch.Tensor
    ) -> torch.Tensor:
        """
        Evaluate the likelihood.

        Evaluates the likelihood of observations given the signals and the
        corresponding gaussian parameters.

        Parameters
        ----------
        observations : torch.Tensor
            Noisy observations. Shape is (batch, 1, dim1, dim2).
        signals : torch.Tensor
            Underlying signals. Shape is (batch, 1, dim1, dim2).

        Returns
        -------
        torch.Tensor
            Likelihood of observations given the signals and the GMM noise model.
        """
        observations = observations.float()
        signals = signals.float()
        gaussian_parameters: list[torch.Tensor] = self.get_gaussian_parameters(signals)
        p = 0  # torch.zeros_like(observations)
        for gaussian in range(self.n_gaussian):
            # Ensure all tensors have compatible shapes
            mean = gaussian_parameters[gaussian]
            std = gaussian_parameters[self.n_gaussian + gaussian]
            weight = gaussian_parameters[2 * self.n_gaussian + gaussian]

            # Compute normal density
            p += (
                self.normal_density(
                    observations,
                    mean,
                    std,
                )
                * weight
            )
        return p + self.tolerance

    def get_gaussian_parameters(self, signals: torch.Tensor) -> list[torch.Tensor]:
        """
        Return the noise model for given signals.

        Parameters
        ----------
        signals : torch.Tensor
            Underlying signals.

        Returns
        -------
        list[torch.Tensor]
            Contains a list of `mu`, `sigma` and `alpha` for the `signals`.
        """
        noise_model = []
        mu = []
        sigma = []
        alpha = []
        kernels = self.weight.shape[0] // 3
        device = signals.device
        self.min_signal = self.min_signal.to(device)
        self.max_signal = self.max_signal.to(device)
        self.min_sigma = self.min_sigma.to(device)
        self.tolerance = self.tolerance.to(device)
        for num in range(kernels):
            mu.append(self.polynomial_regressor(self.weight[num, :], signals))
            expval = torch.exp(self.weight[kernels + num, :])
            sigma_temp = self.polynomial_regressor(expval, signals)
            sigma_temp = torch.clamp(sigma_temp, min=self.min_sigma)
            sigma.append(torch.sqrt(sigma_temp))

            expval = torch.exp(
                self.polynomial_regressor(self.weight[2 * kernels + num, :], signals)
                + self.tolerance
            )
            alpha.append(expval)

        sum_alpha = 0
        for al in range(kernels):
            sum_alpha = alpha[al] + sum_alpha

        # sum of alpha is forced to be 1.
        for ker in range(kernels):
            alpha[ker] = alpha[ker] / sum_alpha

        sum_means = 0
        # sum_means is the alpha weighted average of the means
        for ker in range(kernels):
            sum_means = alpha[ker] * mu[ker] + sum_means

        # subtracting the alpha weighted average of the means from the means
        # ensures that the GMM has the inclination to have the mean=signals.
        # its like a residual conection. I don't understand why we learn the mean?
        for ker in range(kernels):
            mu[ker] = mu[ker] - sum_means + signals

        for i in range(kernels):
            noise_model.append(mu[i])
        for j in range(kernels):
            noise_model.append(sigma[j])
        for k in range(kernels):
            noise_model.append(alpha[k])

        return noise_model

    @staticmethod
    def _fast_shuffle(series: torch.Tensor, num: int) -> torch.Tensor:
        """Shuffle the inputs randomly `num` times.

        Parameters
        ----------
        series : torch.Tensor
            Input tensor to shuffle along the first dimension.
        num : int
            Number of times to shuffle.

        Returns
        -------
        torch.Tensor
            The shuffled tensor.
        """
        length = series.shape[0]
        for _ in range(num):
            idx = torch.randperm(length)
            series = series[idx, :]
        return series

    def get_signal_observation_pairs(
        self,
        signal: NDArray,
        observation: NDArray,
        lower_clip: float,
        upper_clip: float,
    ) -> torch.Tensor:
        """Return the signal-observation pixel intensities as a two-column tensor.

        Parameters
        ----------
        signal : NDArray
            Clean signal data.
        observation : NDArray
            Noisy observation data.
        lower_clip : float
            Lower percentile bound for clipping.
        upper_clip : float
            Upper percentile bound for clipping.

        Returns
        -------
        torch.Tensor
            Shuffled two-column tensor of (signal, observation) pixel intensities.
        """
        lb = np.percentile(signal, lower_clip)
        ub = np.percentile(signal, upper_clip)
        stepsize = observation[0].size
        n_observations = observation.shape[0]
        n_signals = signal.shape[0]
        sig_obs_pairs = np.zeros((n_observations * stepsize, 2))

        for i in range(n_observations):
            j = i // (n_observations // n_signals)
            sig_obs_pairs[stepsize * i : stepsize * (i + 1), 0] = signal[j].ravel()
            sig_obs_pairs[stepsize * i : stepsize * (i + 1), 1] = observation[i].ravel()
        sig_obs_pairs = sig_obs_pairs[
            (sig_obs_pairs[:, 0] > lb) & (sig_obs_pairs[:, 0] < ub)
        ]
        sig_obs_pairs = sig_obs_pairs.astype(np.float32)
        sig_obs_pairs = torch.from_numpy(sig_obs_pairs)
        return self._fast_shuffle(sig_obs_pairs, 2)

    def fit(
        self,
        signal: NDArray,
        observation: NDArray,
        learning_rate: float = 1e-1,
        batch_size: int = 250000,
        n_epochs: int = 2000,
        lower_clip: float = 0.0,
        upper_clip: float = 100.0,
    ) -> list[float]:
        """Train the noise model on signal-observation pairs.

        Parameters
        ----------
        signal : NDArray
            Clean signal data.
        observation : NDArray
            Noisy observation data.
        learning_rate : float
            Learning rate. Default = 1e-1.
        batch_size : int
            Mini-batch size. Default = 250000.
        n_epochs : int
            Number of epochs. Default = 2000.
        lower_clip : float
            Lower percentile for clipping. Default is 0.
        upper_clip : float
            Upper percentile for clipping. Default is 100.

        Returns
        -------
        list[float]
            Training loss for each epoch.
        """
        self._set_model_mode(mode="train")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to_device(device)
        optimizer = torch.optim.Adam([self.weight], lr=learning_rate)

        sig_obs_pairs = self.get_signal_observation_pairs(
            signal, observation, lower_clip, upper_clip
        )

        train_losses = []
        counter = 0
        for t in range(n_epochs):
            if (counter + 1) * batch_size >= sig_obs_pairs.shape[0]:
                counter = 0
                sig_obs_pairs = self._fast_shuffle(sig_obs_pairs, 1)

            batch_vectors = sig_obs_pairs[
                counter * batch_size : (counter + 1) * batch_size, :
            ]
            observations = batch_vectors[:, 1].to(self.device)
            signals = batch_vectors[:, 0].to(self.device)

            p = self.likelihood(observations, signals)

            joint_loss = torch.mean(-torch.log(p))
            train_losses.append(joint_loss.item())

            if self.weight.isnan().any() or self.weight.isinf().any():
                print(
                    "NaN or Inf detected in the weights. Aborting training at epoch: ",
                    t,
                )
                break

            if t % 100 == 0:
                last_losses = train_losses[-100:]
                print(t, np.mean(last_losses))

            optimizer.zero_grad()
            joint_loss.backward()
            optimizer.step()
            counter += 1

        self._set_model_mode(mode="prediction")
        self.to_device(torch.device("cpu"))
        print("===================\n")
        return train_losses

    def sample_observation_from_signal(self, signal: NDArray) -> NDArray:
        """Sample noisy observations from the learned noise model.

        For each pixel in the input signal, samples a corresponding noisy
        pixel from the Gaussian Mixture Model.

        Note: when this model is normalized (`is_normalized` is True), the input
        signal and the returned samples live in normalized data space.

        Parameters
        ----------
        signal : NDArray
            Clean signal data. Can be 2D (Y, X) or higher dimensional.
            For 3D+ arrays, sampling is performed independently for each 2D slice.

        Returns
        -------
        NDArray
            Sampled noisy observation with same shape as input signal.
        """
        if signal.ndim < 2:
            raise ValueError(f"Signal must be at least 2D, got {signal.ndim}D")

        if signal.ndim == 2:
            return self._sample_2d(signal)

        original_shape = signal.shape
        flat_signal = signal.reshape(-1, *signal.shape[-2:])
        samples = np.stack([self._sample_2d(s) for s in flat_signal], axis=0)
        return samples.reshape(original_shape)

    def _sample_2d(self, signal: NDArray) -> NDArray:
        """Sample noisy observation for a single 2D image.

        Parameters
        ----------
        signal : NDArray
            Clean 2D signal data with shape (Y, X).

        Returns
        -------
        NDArray
            Sampled noisy observation with shape (Y, X).
        """
        signal_tensor = torch.from_numpy(signal).to(torch.float32)
        height, width = signal_tensor.shape

        with torch.no_grad():
            gaussian_params = self.get_gaussian_parameters(signal_tensor)
            means = np.array(gaussian_params[: self.n_gaussian])
            stds = np.array(gaussian_params[self.n_gaussian : self.n_gaussian * 2])
            alphas = np.array(gaussian_params[self.n_gaussian * 2 :])

            if self.n_gaussian == 1:
                observation = np.random.normal(
                    loc=means[0], scale=stds[0], size=(height, width)
                )
            else:
                uniform = np.random.rand(1, height, width)
                cumulative_alphas = np.cumsum(alphas, axis=0)
                selected_component = np.argmax(
                    uniform < cumulative_alphas, axis=0, keepdims=True
                )

                selected_mus = np.take_along_axis(means, selected_component, axis=0)
                selected_stds = np.take_along_axis(stds, selected_component, axis=0)
                selected_mus = selected_mus.squeeze(0)
                selected_stds = selected_stds.squeeze(0)

                observation = np.random.normal(
                    selected_mus, selected_stds, size=(height, width)
                )
        return observation

    def save(self, path: str, name: str, channel_index: int | None = None) -> None:
        """Save the trained parameters on the noise model.

        Parameters
        ----------
        path : str
            Path to save the trained parameters.
        name : str
            File name to save the trained parameters.
        channel_index : int | None, optional
            The data channel index this model was trained on.  When provided it
            is stored in the `.npz` file so channel ordering can be validated on
            load.

        Raises
        ------
        ValueError
            If this model is normalized: the `.npz` format stores raw-space
            models only.
        """
        if self.is_normalized:
            raise ValueError(
                "Refusing to save a normalized noise model: the .npz format "
                "stores raw-space models only. Save the raw model and normalize "
                "at load time with `get_normalized_copy`."
            )
        os.makedirs(path, exist_ok=True)
        save_kwargs: dict = {
            "trained_weight": self.weight.numpy(),
            "min_signal": self.min_signal.numpy(),
            "max_signal": self.max_signal.numpy(),
            "min_sigma": self.min_sigma,
        }
        if channel_index is not None:
            save_kwargs["channel_index"] = np.array(channel_index)
        np.savez(os.path.join(path, name), **save_kwargs)
        print("The trained parameters (" + name + ") is saved at location: " + path)
