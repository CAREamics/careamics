from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray

if TYPE_CHECKING:
    from careamics.config import GaussianMixtureNMConfig, MultiChannelNMConfig

# TODO this module shouldn't be in lvae folder


def create_histogram(
    bins: int, min_val: float, max_val: float, observation: NDArray, signal: NDArray
) -> NDArray:
    """
    Creates a 2D histogram from 'observation' and 'signal'.

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
    histogram : np.ndarray
        A 3D array:
        - histogram[0]: Normalized 2D counts.
        - histogram[1]: Lower boundaries of bins along y.
        - histogram[2]: Upper boundaries of bins along y.
    The values for x can be obtained by transposing 'histogram[1]' and 'histogram[2]'.
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
    model_config: Optional[MultiChannelNMConfig],
) -> Optional[MultiChannelNoiseModel]:
    """Noise model factory.

    Parameters
    ----------
    model_config : Optional[MultiChannelNMConfig]
        Noise model configuration, a `MultiChannelNMConfig` config that defines
        noise models for the different output channels.

    Returns
    -------
    Optional[MultiChannelNoiseModel]
        A noise model instance.

    Raises
    ------
    NotImplementedError
        If the chosen noise model `model_type` is not implemented.
        Currently only `GaussianMixtureNoiseModel` is implemented.
    """
    if model_config:
        noise_models = []
        for nm in model_config.noise_models:
            if nm.path:
                if nm.model_type == "GaussianMixtureNoiseModel":
                    noise_models.append(GaussianMixtureNoiseModel(nm))
                else:
                    raise NotImplementedError(
                        f"Model {nm.model_type} is not implemented"
                    )

            else:  # TODO this means signal/obs are provided. Controlled in pydantic model
                # TODO train a new model. Config should always be provided?
                if nm.model_type == "GaussianMixtureNoiseModel":
                    # TODO one model for each channel all make this choise inside the model?
                    # trained_nm = train_gm_noise_model(nm)
                    # noise_models.append(trained_nm)
                    raise NotImplementedError(
                        "GaussianMixtureNoiseModel model training is not implemented."
                    )
                else:
                    raise NotImplementedError(
                        f"Model {nm.model_type} is not implemented"
                    )
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
    model_config : GaussianMixtureNoiseModel
        _description_

    Returns
    -------
    _description_
    """
    # TODO where to put train params?
    # TODO any training params ? Different channels ?
    noise_model = GaussianMixtureNoiseModel(model_config)
    # TODO revisit config unpacking
    noise_model.fit(signal, observation)
    return noise_model


class MultiChannelNoiseModel(nn.Module):
    def __init__(self, nmodels: list[GaussianMixtureNoiseModel]):
        """Constructor.

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
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def to_device(self, device: torch.device):
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


class GaussianMixtureNoiseModel(nn.Module):
    """Define a noise model parameterized as a mixture of gaussians.

    If `config.path` is not provided a new object is initialized from scratch.
    Otherwise, a model is loaded from `config.path`.

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
    path: Union[str, Path]
        Path to the directory where the trained noise model (*.npz) is saved in the `train` method.
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
        parameters and the signal. 2 implies a linear relationship, 3 implies a quadratic
        relationship and so on.
    device: device
        GPU device.
    min_sigma: float
        All values of `standard deviation` below this are clamped to this value.
    """

    # TODO training a NM relies on getting a clean data(N2V e.g,)
    def __init__(self, config: GaussianMixtureNMConfig) -> None:
        super().__init__()
        self.device = torch.device("cpu")

        if config.path is not None:
            params = np.load(config.path)
        else:
            params = config.model_dump(exclude_none=True)

        min_sigma = torch.tensor(params["min_sigma"])
        min_signal = torch.tensor(params["min_signal"])
        max_signal = torch.tensor(params["max_signal"])
        self.register_buffer("min_signal", min_signal)
        self.register_buffer("max_signal", max_signal)
        self.register_buffer("min_sigma", min_sigma)
        self.register_buffer("tolerance", torch.tensor([1e-10]))

        if "trained_weight" in params:
            weight = torch.tensor(params["trained_weight"])
        elif "weight" in params and params["weight"] is not None:
            weight = torch.tensor(params["weight"])
        else:
            weight = self._initialize_weights(
                params["n_gaussian"], params["n_coeff"], max_signal, min_signal
            )

        self.n_gaussian = weight.shape[0] // 3
        self.n_coeff = weight.shape[1]

        self.register_parameter("weight", nn.Parameter(weight))
        self._set_model_mode(mode="prediction")

        print(f"[{self.__class__.__name__}] min_sigma: {self.min_sigma}")

    def _initialize_weights(
        self,
        n_gaussian: int,
        n_coeff: int,
        max_signal: torch.Tensor,
        min_signal: torch.Tensor,
    ) -> torch.Tensor:
        """Create random weight initialization."""
        weight = torch.randn(n_gaussian * 3, n_coeff)
        weight[n_gaussian : 2 * n_gaussian, 1] = torch.log(
            max_signal - min_signal
        ).float()
        return weight

    def to_device(self, device: torch.device):
        self.device = device
        self.to(device)

    def _set_model_mode(self, mode: str) -> None:
        """Move parameters to the device and set weights' requires_grad depending on the mode"""
        if mode == "train":
            self.weight.requires_grad = True
        else:
            self.weight.requires_grad = False

    def polynomial_regressor(
        self, weight_params: torch.Tensor, signals: torch.Tensor
    ) -> torch.Tensor:
        """Combines `weight_params` and signal `signals` to regress for the gaussian parameter values.

        Parameters
        ----------
        weight_params : Tensor
            Corresponds to specific rows of the `self.weight`

        signals : Tensor
            Signals

        Returns
        -------
        value : Tensor
            Corresponds to either of mean, standard deviation or weight, evaluated at `signals`
        """
        value = torch.zeros_like(signals)
        for i in range(weight_params.shape[0]):
            value += weight_params[i] * (
                ((signals - self.min_signal) / (self.max_signal - self.min_signal)) ** i
            )
        return value

    def normal_density(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        """
        Evaluates the normal probability density at `x` given the mean `mean` and standard deviation `std`.

        Parameters
        ----------
        x: torch.Tensor
            The ground-truth tensor. Shape is (batch, 1, dim1, dim2).
        mean: torch.Tensor
            The inferred mean of distribution. Shape is (batch, 1, dim1, dim2).
        std: torch.Tensor
            The inferred standard deviation of distribution. Shape is (batch, 1, dim1, dim2).

        Returns
        -------
        tmp: torch.Tensor
            Normal probability density of `x` given `mean` and `std`
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
        Evaluates the likelihood of observations given the signals and the corresponding gaussian parameters.

        Parameters
        ----------
        observations : Tensor
            Noisy observations. Shape is (batch, 1, dim1, dim2).
        signals : Tensor
            Underlying signals. Shape is (batch, 1, dim1, dim2).

        Returns
        -------
        value: torch.Tensor:
            Likelihood of observations given the signals and the GMM noise model
        """
        gaussian_parameters: list[torch.Tensor] = self.get_gaussian_parameters(signals)
        p = torch.zeros_like(observations)
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
        Returns the noise model for given signals

        Parameters
        ----------
        signals : Tensor
            Underlying signals

        Returns
        -------
        noise_model: list of Tensor
            Contains a list of `mu`, `sigma` and `alpha` for the `signals`
        """
        noise_model = []
        mu = []
        sigma = []
        alpha = []
        kernels = self.weight.shape[0] // 3
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
        # its like a residual conection. I don't understand why we need to learn the mean?
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
        """Shuffle the inputs randomly num times"""
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
        """Returns the Signal-Observation pixel intensities as a two-column array

        Parameters
        ----------
        signal : numpy array
            Clean Signal Data
        observation: numpy array
            Noisy observation Data
        lower_clip: float
            Lower percentile bound for clipping.
        upper_clip: float
            Upper percentile bound for clipping.

        Returns
        -------
        noise_model: list of torch floats
            Contains a list of `mu`, `sigma` and `alpha` for the `signals`
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
        """Training to learn the noise model from signal - observation pairs.

        Parameters
        ----------
        signal: numpy array
            Clean Signal Data
        observation: numpy array
            Noisy Observation Data
        learning_rate: float
            Learning rate. Default = 1e-1.
        batch_size: int
            Nini-batch size. Default = 250000.
        n_epochs: int
            Number of epochs. Default = 2000.
        lower_clip : int
            Lower percentile for clipping. Default is 0.
        upper_clip : int
            Upper percentile for clipping. Default is 100.
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
        """
        Sample an instance of observation based on an input signal using a
        learned Gaussian Mixture Model. For each pixel in the input signal,
        samples a corresponding noisy pixel.

        Parameters
        ----------
        signal: numpy array
            Clean 2D signal data.

        Returns
        -------
        observation: numpy array
            An instance of noisy observation data based on the input signal.
        """
        assert len(signal.shape) == 2, "Only 2D inputs are supported."

        signal_tensor = torch.from_numpy(signal).to(torch.float32)
        height, width = signal_tensor.shape

        with torch.no_grad():
            # Get gaussian parameters for each pixel
            gaussian_params = self.get_gaussian_parameters(signal_tensor)
            means = np.array(gaussian_params[: self.n_gaussian])
            stds = np.array(gaussian_params[self.n_gaussian : self.n_gaussian * 2])
            alphas = np.array(gaussian_params[self.n_gaussian * 2 :])

            if self.n_gaussian == 1:
                # Single gaussian case
                observation = np.random.normal(
                    loc=means[0], scale=stds[0], size=(height, width)
                )
            else:
                # Multiple gaussians: sample component for each pixel
                uniform = np.random.rand(1, height, width)
                # Compute cumulative probabilities for component selection
                cumulative_alphas = np.cumsum(
                    alphas, axis=0
                )  # Shape: (n_gaussian, height, width)
                selected_component = np.argmax(
                    uniform < cumulative_alphas, axis=0, keepdims=True
                )

                # For every pixel, choose the corresponding gaussian
                # and get the learned mu and sigma
                selected_mus = np.take_along_axis(means, selected_component, axis=0)
                selected_stds = np.take_along_axis(stds, selected_component, axis=0)
                selected_mus = selected_mus.squeeze(0)
                selected_stds = selected_stds.squeeze(0)

                # Sample from the normal distribution with learned mu and sigma
                observation = np.random.normal(
                    selected_mus, selected_stds, size=(height, width)
                )
        return observation

    def save(self, path: str, name: str) -> None:
        """Save the trained parameters on the noise model.

        Parameters
        ----------
        path : str
            Path to save the trained parameters.
        name : str
            File name to save the trained parameters.
        """
        os.makedirs(path, exist_ok=True)
        np.savez(
            os.path.join(path, name),
            trained_weight=self.weight.numpy(),
            min_signal=self.min_signal.numpy(),
            max_signal=self.max_signal.numpy(),
            min_sigma=self.min_sigma,
        )
        print("The trained parameters (" + name + ") is saved at location: " + path)
