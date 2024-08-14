import json
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

from careamics.config import GaussianMixtureNmModel

from .utils import ModelType

# TODO this module shouldn't be in lvae folder


def noise_model_factory(
    model_config: Union[GaussianMixtureNmModel, None],
    paths: Optional[list[Union[str, Path]]] = None,
) -> nn.Module:
    """Noise model factory.

    Parameters
    ----------
    model_config : GaussianMixtureNoiseModel
        _description_

    Returns
    -------
    nn.Module
        _description_

    Raises
    ------
    NotImplementedError
        _description_
    """
    # TODO in case path are provided, load. Should they be in the config ?
    if paths:
        if model_config.model_type == "GaussianMixtureNoiseModel":
            noise_models = []
            for path in paths:
                model_config.path = path
                noise_models.append(GaussianMixtureNoiseModel(model_config))
            return DisentNoiseModel(*noise_models)
        else:
            raise NotImplementedError(
                f"Model {model_config.model_type} is not implemented"
            )
    # TODO should config and paths be mutually exclusive ?
    else:
        # TODO train a new model. Config should always be provided?
        if model_config.model_type == "GaussianMixtureNoiseModel":
            # TODO one model for each channel all make this choise inside the model?
            return train_gaussian_mixture_noise_model(model_config)
    return None


def train_gaussian_mixture_noise_model(model_config: GaussianMixtureNmModel):
    """Train a Gaussian mixture noise model.

    Parameters
    ----------
    model_config : GaussianMixtureNoiseModel
        _description_

    Returns
    -------
    _description_
    """
    # TODO pseudocode
    """
    noise_models = []
    # TODO any training params ? Different channels ?
    for ch in channels:
        noise_model = GaussianMixtureNoiseModel(model_config)
        noise_model.train()
        noise_models.append(noise_model)
    """
    return DisentNoiseModel(*noise_models)


def get_noise_model(
    enable_noise_model: bool,
    model_type: ModelType,
    noise_model_type: str,
    noise_model_ch1_fpath: str,
    noise_model_ch2_fpath: str,
    noise_model_learnable: bool = False,
    denoise_channel: str = "input",  # TODO hardcoded ?
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
                # TODO WTF are input and ch1 the same ?
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


class DisentNoiseModel(nn.Module):
    # TODO this class is a mess(this is an autogenerated btw). But really wtf?!?!?!
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
        # TODO shapes ? names? wtf ?
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

    # TODO training a NM relies on getting a clean data(N2V e.g,)
    def __init__(self, config: GaussianMixtureNmModel):
        super().__init__()
        self._learnable = False

        if config.path is None:
            # TODO is this to train a nm ?
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
        """Returns the noise model for given signals.

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

    def forward(self, x, y):
        """THIS IS A DUMMY METHOD BECAUSE THIS WHOLE MODULE IS A TOTAL MESS."""
        return x, y

    # TODO taken from pn2v. Ashesh needs to clarify this
    def train(self, signal, observation, learning_rate=1e-1, batchSize=250000, n_epochs=2000, name= 'GMMNoiseModel.npz', lowerClip=0, upperClip=100):
        """Training to learn the noise model from signal - observation pairs.

        Parameters
        ----------
        signal: numpy array
            Clean Signal Data
        observation: numpy array
            Noisy Observation Data
        learning_rate: float
            Learning rate. Default = 1e-1.
        batchSize: int
            Nini-batch size. Default = 250000.
        n_epochs: int
            Number of epochs. Default = 2000.
        name: string

            Model name. Default is `GMMNoiseModel`. This model after being trained is saved at the location `path`.

        lowerClip : int
            Lower percentile for clipping. Default is 0.
        upperClip : int
            Upper percentile for clipping. Default is 100.


        """
        sig_obs_pairs=self.getSignalObservationPairs(signal, observation, lowerClip, upperClip)
        counter=0
        optimizer = torch.optim.Adam([self.weight], lr=learning_rate)
        for t in range(n_epochs):

            jointLoss=0
            if (counter+1)*batchSize >= sig_obs_pairs.shape[0]:
                counter=0
                sig_obs_pairs=fastShuffle(sig_obs_pairs,1)

            batch_vectors = sig_obs_pairs[counter*batchSize:(counter+1)*batchSize, :]
            observations = batch_vectors[:,1].astype(np.float32)
            signals = batch_vectors[:,0].astype(np.float32)
            observations = torch.from_numpy(observations.astype(np.float32)).float().to(self.device)
            signals = torch.from_numpy(signals).float().to(self.device)
            p = self.likelihood(observations, signals)
            loss=torch.mean(-torch.log(p))
            jointLoss=jointLoss+loss

            if t%100==0:
                print(t, jointLoss.item())


            if (t%(int(n_epochs*0.5))==0):
                trained_weight = self.weight.cpu().detach().numpy()
                min_signal = self.min_signal.cpu().detach().numpy()
                max_signal = self.max_signal.cpu().detach().numpy()
                np.savez(self.path+name, trained_weight=trained_weight, min_signal = min_signal, max_signal = max_signal, min_sigma = self.min_sigma)




            optimizer.zero_grad()
            jointLoss.backward()
            optimizer.step()
            counter+=1

        print("===================\n")
        print("The trained parameters (" + name + ") is saved at location: "+ self.path)
