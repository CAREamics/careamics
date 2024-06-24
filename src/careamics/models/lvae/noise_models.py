import json
import os

import numpy as np
import torch
import torch.nn as nn

from .utils import ModelType


class DisentNoiseModel(nn.Module):

    def __init__(self, *nmodels):
        """
        Constructor.

        This class receives as input a variable number of noise models, each one corresponding to a channel.
        """
        super().__init__()
        # self.nmodels = nmodels
        for i, nmodel in enumerate(nmodels):
            if nmodel is not None:
                self.add_module(f"nmodel_{i}", nmodel)

        self._nm_cnt = 0
        for nmodel in nmodels:
            if nmodel is not None:
                self._nm_cnt += 1

        print(f"[{self.__class__.__name__}] Nmodels count:{self._nm_cnt}")

    def likelihood(self, obs: torch.Tensor, signal: torch.Tensor) -> torch.Tensor:

        if obs.shape[1] == 1:
            assert signal.shape[1] == 1
            assert self.n2model is None
            return self.nmodel_0.likelihood(obs, signal)

        assert obs.shape[1] == self._nm_cnt, f"{obs.shape[1]} != {self._nm_cnt}"

        ll_list = []
        for ch_idx in range(obs.shape[1]):
            nmodel = getattr(self, f"nmodel_{ch_idx}")
            ll_list.append(
                nmodel.likelihood(
                    obs[:, ch_idx : ch_idx + 1], signal[:, ch_idx : ch_idx + 1]
                )
            )

        return torch.cat(ll_list, dim=1)


def last2path(fpath: str):
    return os.path.join(*fpath.split("/")[-2:])


def get_nm_config(noise_model_fpath: str):
    config_fpath = os.path.join(os.path.dirname(noise_model_fpath), "config.json")
    with open(config_fpath) as f:
        noise_model_config = json.load(f)
    return noise_model_config


def fastShuffle(series, num):
    length = series.shape[0]
    for i in range(num):
        series = series[np.random.permutation(length), :]
    return series


def get_noise_model(
    enable_noise_model: bool,
    model_type: ModelType,
    noise_model_type: str,
    noise_model_ch1_fpath: str,
    noise_model_ch2_fpath: str,
    noise_model_learnable: bool = False,
    denoise_channel: str = "input",
):
    if enable_noise_model:
        nmodels = []
        # HDN -> one single output -> one single noise model
        if model_type == ModelType.Denoiser:
            if noise_model_type == "hist":
                raise NotImplementedError(
                    '"hist" noise model is not supported for now.'
                )
            elif noise_model_type == "gmm":
                if denoise_channel == "Ch1":
                    nmodel_fpath = noise_model_ch1_fpath
                    print(f"Noise model Ch1: {nmodel_fpath}")
                    nmodel1 = GaussianMixtureNoiseModel(params=np.load(nmodel_fpath))
                    nmodel2 = None
                    nmodels = [nmodel1, nmodel2]
                elif denoise_channel == "Ch2":
                    nmodel_fpath = noise_model_ch2_fpath
                    print(f"Noise model Ch2: {nmodel_fpath}")
                    nmodel1 = GaussianMixtureNoiseModel(params=np.load(nmodel_fpath))
                    nmodel2 = None
                    nmodels = [nmodel1, nmodel2]
                elif denoise_channel == "input":
                    nmodel_fpath = noise_model_ch1_fpath
                    print(f"Noise model input: {nmodel_fpath}")
                    nmodel1 = GaussianMixtureNoiseModel(params=np.load(nmodel_fpath))
                    nmodel2 = None
                    nmodels = [nmodel1, nmodel2]
                else:
                    raise ValueError(f"Invalid denoise_channel: {denoise_channel}")
        # muSplit -> two outputs -> two noise models
        elif noise_model_type == "gmm":
            print(f"Noise model Ch1: {noise_model_ch1_fpath}")
            print(f"Noise model Ch2: {noise_model_ch2_fpath}")

            nmodel1 = GaussianMixtureNoiseModel(params=np.load(noise_model_ch1_fpath))
            nmodel2 = GaussianMixtureNoiseModel(params=np.load(noise_model_ch2_fpath))

            nmodels = [nmodel1, nmodel2]

            # if 'noise_model_ch3_fpath' in config.model:
            #     print(f'Noise model Ch3: {config.model.noise_model_ch3_fpath}')
            #     nmodel3 = GaussianMixtureNoiseModel(params=np.load(config.model.noise_model_ch3_fpath))
            #     nmodels = [nmodel1, nmodel2, nmodel3]
            # else:
            #     nmodels = [nmodel1, nmodel2]
        else:
            raise ValueError(f"Invalid noise_model_type: {noise_model_type}")

        if noise_model_learnable:
            for nmodel in nmodels:
                if nmodel is not None:
                    nmodel.make_learnable()

        return DisentNoiseModel(*nmodels)
    return None


class GaussianMixtureNoiseModel(nn.Module):
    """
    The GaussianMixtureNoiseModel class describes a noise model which is parameterized as a mixture of gaussians.
    If you would like to initialize a new object from scratch, then set `params`= None and specify the other parameters as keyword arguments.
    If you are instead loading a model, use only `params`.

    Parameters
    ----------
    **kwargs: keyworded, variable-length argument dictionary.
    Arguments include:
        min_signal : float
            Minimum signal intensity expected in the image.
        max_signal : float
            Maximum signal intensity expected in the image.
        path: string
            Path to the directory where the trained noise model (*.npz) is saved in the `train` method.
        weight : array
            A [3*n_gaussian, n_coeff] sized array containing the values of the weights describing the noise model.
            Each gaussian contributes three parameters (mean, standard deviation and weight), hence the number of rows in `weight` are 3*n_gaussian.
            If `weight=None`, the weight array is initialized using the `min_signal` and `max_signal` parameters.
        n_gaussian: int
            Number of gaussians.
        n_coeff: int
            Number of coefficients to describe the functional relationship between gaussian parameters and the signal.
            2 implies a linear relationship, 3 implies a quadratic relationship and so on.
        device: device
            GPU device.
        min_sigma: int
            All values of sigma (`standard deviation`) below min_sigma are clamped to become equal to min_sigma.
        params: dictionary
            Use `params` if one wishes to load a model with trained weights.
            While initializing a new object of the class `GaussianMixtureNoiseModel` from scratch, set this to `None`.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self._learnable = False

        if kwargs.get("params") is None:
            weight = kwargs.get("weight")
            n_gaussian = kwargs.get("n_gaussian")
            n_coeff = kwargs.get("n_coeff")
            min_signal = kwargs.get("min_signal")
            max_signal = kwargs.get("max_signal")
            # self.device = kwargs.get('device')
            self.path = kwargs.get("path")
            self.min_sigma = kwargs.get("min_sigma")
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
            params = kwargs.get("params")
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

        #

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

    def normalDens(self, x, m_=0.0, std_=None):
        """Evaluates the normal probability density at `x` given the mean `m` and standard deviation `std`.

        Parameters
        ----------
        x: torch.cuda.FloatTensor
            Observations
        m_: torch.cuda.FloatTensor
            Mean
        std_: torch.cuda.FloatTensor
            Standard-deviation

        Returns
        -------
        tmp: torch.cuda.FloatTensor
            Normal probability density of `x` given `m_` and `std_`
        """
        tmp = -((x - m_) ** 2)
        tmp = tmp / (2.0 * std_ * std_)
        tmp = torch.exp(tmp)
        tmp = tmp / torch.sqrt((2.0 * np.pi) * std_ * std_)
        return tmp

    def likelihood(self, observations, signals):
        """Evaluates the likelihood of observations given the signals and the corresponding gaussian parameters.

        Parameters
        ----------
        observations : torch.cuda.FloatTensor
            Noisy observations
        signals : torch.cuda.FloatTensor
            Underlying signals

        Returns
        -------
        value :p + self.tol
            Likelihood of observations given the signals and the GMM noise model
        """
        self.to_device(signals)
        gaussianParameters = self.getGaussianParameters(signals)
        p = 0
        for gaussian in range(self.n_gaussian):
            p += (
                self.normalDens(
                    observations,
                    gaussianParameters[gaussian],
                    gaussianParameters[self.n_gaussian + gaussian],
                )
                * gaussianParameters[2 * self.n_gaussian + gaussian]
            )
        return p + self.tol

    def getGaussianParameters(self, signals):
        """Returns the noise model for given signals

        Parameters
        ----------
        signals : torch.cuda.FloatTensor
            Underlying signals

        Returns
        -------
        noiseModel: list of torch.cuda.FloatTensor
            Contains a list of `mu`, `sigma` and `alpha` for the `signals`

        """
        noiseModel = []
        mu = []
        sigma = []
        alpha = []
        kernels = self.weight.shape[0] // 3
        for num in range(kernels):
            mu.append(self.polynomialRegressor(self.weight[num, :], signals))
            # expval = torch.exp(torch.clamp(self.weight[kernels + num, :], max=MAX_VAR_W))
            expval = torch.exp(self.weight[kernels + num, :])
            # self.maxval = max(self.maxval, expval.max().item())
            sigmaTemp = self.polynomialRegressor(expval, signals)
            sigmaTemp = torch.clamp(sigmaTemp, min=self.min_sigma)
            sigma.append(torch.sqrt(sigmaTemp))

            # expval = torch.exp(
            #     torch.clamp(
            #         self.polynomialRegressor(self.weight[2 * kernels + num, :], signals) + self.tol, MAX_ALPHA_W))
            expval = torch.exp(
                self.polynomialRegressor(self.weight[2 * kernels + num, :], signals)
                + self.tol
            )
            # self.maxval = max(self.maxval, expval.max().item())
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

        mu_shifted = []
        # subtracting the alpha weighted average of the means from the means
        # ensures that the GMM has the inclination to have the mean=signals.
        # its like a residual conection. I don't understand why we need to learn the mean?
        for ker in range(kernels):
            mu[ker] = mu[ker] - sum_means + signals

        for i in range(kernels):
            noiseModel.append(mu[i])
        for j in range(kernels):
            noiseModel.append(sigma[j])
        for k in range(kernels):
            noiseModel.append(alpha[k])

        return noiseModel

    def getSignalObservationPairs(self, signal, observation, lowerClip, upperClip):
        """Returns the Signal-Observation pixel intensities as a two-column array

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
        noiseModel: list of torch floats
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
