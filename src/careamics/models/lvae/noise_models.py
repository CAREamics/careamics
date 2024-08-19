from __future__ import annotations

from typing import Union, TYPE_CHECKING, Optional

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from careamics.config import GaussianMixtureNmModel, MultiChannelNmModel

# TODO this module shouldn't be in lvae folder


def noise_model_factory(
    model_config: Optional[MultiChannelNmModel],
) -> Optional[MultiChannelNoiseModel]:
    """Noise model factory.

    Parameters
    ----------
    model_config : Optional[MultiChannelNmModel]
        Noise model configuration, a `MultiChannelNmModel` config that defines
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
        for nm_config in model_config.noise_models:
            if nm_config.model_type == "GaussianMixtureNoiseModel":
                noise_models.append(GaussianMixtureNoiseModel(nm_config))
            else:
                raise NotImplementedError(
                    f"Model {nm_config.model_type} is not implemented"
                )
        return MultiChannelNoiseModel(noise_models)
    # TODO should config and paths be mutually exclusive? What you mean, Igor?
    return None


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
        for i, nmodel in enumerate(nmodels):
            if nmodel is not None:
                self.add_module(
                    f"nmodel_{i}", nmodel
                )  # TODO: wouldn't be easier to use a list?

        self._nm_cnt = 0
        for nmodel in nmodels:
            if nmodel is not None:
                self._nm_cnt += 1

        print(f"[{self.__class__.__name__}] Nmodels count:{self._nm_cnt}")

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


# TODO: is this needed?
def fastShuffle(series, num):
    length = series.shape[0]
    for i in range(num):
        series = series[np.random.permutation(length), :]
    return series


class GaussianMixtureNoiseModel(nn.Module):
    """Define a noise model parameterized as a mixture of gaussians.

    If `config.path` is not provided a new object is initialized from scratch.
    Otherwise, a model is loaded from `config.path`.

    Parameters
    ----------
    config : GaussianMixtureNmModel
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
    def __init__(self, config: GaussianMixtureNmModel):
        super().__init__()
        self._learnable = False

        if config.path is None:
            # TODO this is (probably) to train a nm. We leave it for later refactoring
            weight = config.weight
            n_gaussian = config.n_gaussian
            n_coeff = config.n_coeff
            min_signal = config.min_signal
            max_signal = config.max_signal
            # self.device = kwargs.get('device')
            self.min_sigma = config.min_sigma
            if weight is None:
                weight = np.random.randn(n_gaussian * 3, n_coeff)
                weight[n_gaussian : 2 * n_gaussian, 1] = np.log(max_signal - min_signal)
                weight = torch.from_numpy(
                    weight.astype(np.float32)
                ).float()  # .to(self.device)
                weight = nn.Parameter(weight, requires_grad=True)

            self.n_gaussian = weight.shape[0] // 3
            self.n_coeff = weight.shape[1]
            self.weight = weight
            self.min_signal = torch.Tensor([min_signal])  # .to(self.device)
            self.max_signal = torch.Tensor([max_signal])  # .to(self.device)
            self.tol = torch.Tensor([1e-10])  # .to(self.device)
        else:
            params = np.load(config.path)
            # self.device = kwargs.get('device')

            self.min_signal = torch.Tensor(params["min_signal"])  # .to(self.device)
            self.max_signal = torch.Tensor(params["max_signal"])  # .to(self.device)

            self.weight = torch.nn.Parameter(
                torch.Tensor(params["trained_weight"]), requires_grad=False
            )  # .to(self.device)
            self.min_sigma = params["min_sigma"].item()
            self.n_gaussian = self.weight.shape[0] // 3
            self.n_coeff = self.weight.shape[1]
            self.tol = torch.Tensor([1e-10])  # .to(self.device)
            self.min_signal = torch.Tensor([self.min_signal])  # .to(self.device)
            self.max_signal = torch.Tensor([self.max_signal])  # .to(self.device)

        print(f"[{self.__class__.__name__}] min_sigma: {self.min_sigma}")

    def make_learnable(self):
        print(f"[{self.__class__.__name__}] Making noise model learnable")
        self._learnable = True
        self.weight.requires_grad = True

    def to_device(self, cuda_tensor):
        # move everything to GPU
        if self.min_signal.device != cuda_tensor.device:
            self.max_signal = self.max_signal.to(cuda_tensor.device)
            self.min_signal = self.min_signal.to(cuda_tensor.device)
            self.tol = self.tol.to(cuda_tensor.device)
            self.weight = self.weight.to(cuda_tensor.device)
            if self._learnable:
                self.weight.requires_grad = True

    def polynomialRegressor(self, weightParams, signals):
        """Combines `weightParams` and signal `signals` to regress for the gaussian parameter values.

        Parameters
        ----------
        weightParams : torch.cuda.FloatTensor
            Corresponds to specific rows of the `self.weight`
        signals : torch.cuda.FloatTensor
            Signals

        Returns
        -------
        value : torch.cuda.FloatTensor
            Corresponds to either of mean, standard deviation or weight, evaluated at `signals`
        """
        value = 0
        for i in range(weightParams.shape[0]):
            value += weightParams[i] * (
                ((signals - self.min_signal) / (self.max_signal - self.min_signal)) ** i
            )
        return value

    def normalDens(
        self, x: torch.Tensor, m_: torch.Tensor = 0.0, std_: torch.Tensor = None
    ) -> torch.Tensor:
        """Evaluates the normal probability density at `x` given the mean `m` and
        standard deviation `std`.

        Parameters
        ----------
        x: torch.Tensor
            Observations (i.e., noisy image).
        m_: torch.Tensor
            Pixel-wise mean.
        std_: torch.Tensor
            Pixel-wise standard deviation.

        Returns
        -------
        tmp: torch.Tensor
            Normal probability density of `x` given `m_` and `std_`
        """
        tmp = -((x - m_) ** 2)
        tmp = tmp / (2.0 * std_ * std_)
        tmp = torch.exp(tmp)
        tmp = tmp / torch.sqrt((2.0 * np.pi) * std_ * std_)
        return tmp

    def likelihood(
        self, observations: torch.Tensor, signals: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate the likelihood of observations given the signals and the
        corresponding gaussian parameters.

        Parameters
        ----------
        observations : torch.cuda.FloatTensor
            Noisy observations.
        signals : torch.cuda.FloatTensor
            Underlying signals.

        Returns
        -------
        value :p + self.tol
            Likelihood of observations given the signals and the GMM noise model
        """
        self.to_device(signals)  # move al needed stuff to the same device as `signals``
        gaussianParameters = self.getGaussianParameters(signals)
        p = 0
        for gaussian in range(self.n_gaussian):
            p += (
                self.normalDens(
                    x=observations,
                    m_=gaussianParameters[gaussian],
                    std_=gaussianParameters[self.n_gaussian + gaussian],
                )
                * gaussianParameters[2 * self.n_gaussian + gaussian]
            )
        return p + self.tol

    def getGaussianParameters(self, signals: torch.Tensor) -> list[torch.Tensor]:
        """Returns the noise model for given signals.

        Parameters
        ----------
        signals : torch.Tensor
            Underlying signals

        Returns
        -------
        gmmParams: list[torch.Tensor]
            A list containing tensors representing `mu`, `sigma` and `alpha`
            parameters for the `n_gaussian` gaussians in the mixture.

        """
        gmmParams = []
        mu = []
        sigma = []
        alpha = []
        kernels = self.weight.shape[0] // 3
        for num in range(kernels):
            # For each Gaussian in the mixture, evaluate mean, std and weight
            mu.append(self.polynomialRegressor(self.weight[num, :], signals))

            expval = torch.exp(self.weight[kernels + num, :])
            # TODO: why taking the exp? it is not in PPN2V paper...
            sigmaTemp = self.polynomialRegressor(expval, signals)
            sigmaTemp = torch.clamp(sigmaTemp, min=self.min_sigma)
            sigma.append(torch.sqrt(sigmaTemp))

            expval = torch.exp(
                self.polynomialRegressor(self.weight[2 * kernels + num, :], signals)
                + self.tol
            )
            alpha.append(expval)  # NOTE: these are the numerators of weights

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
        # TODO: I don't understand why we need to learn the mean?
        for ker in range(kernels):
            mu[ker] = mu[ker] - sum_means + signals

        for i in range(kernels):
            gmmParams.append(mu[i])
        for j in range(kernels):
            gmmParams.append(sigma[j])
        for k in range(kernels):
            gmmParams.append(alpha[k])

        return gmmParams

    # TODO: this is to train the noise model
    def getSignalObservationPairs(self, signal, observation, lowerClip, upperClip):
        """Returns the Signal-Observation pixel intensities as a two-column array.

        Parameters
        ----------
        signal : numpy array
            Clean Signal Data
        observation: numpy array
            Noisy observation Data
        lowerClip: float
            Lower percentile bound for clipping.
        upperClip: float
            Upper percentile bound for clipping.

        Returns
        -------
        gmmParams: list of torch floats
            Contains a list of `mu`, `sigma` and `alpha` for the `signals`
        """
        lb = np.percentile(signal, lowerClip)
        ub = np.percentile(signal, upperClip)
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
        return fastShuffle(sig_obs_pairs, 2)

    # TODO: what's the use of this method?
    def forward(self, x, y):
        """Temporary dummy forward method."""
        return x, y
