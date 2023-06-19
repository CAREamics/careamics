import numpy as np
import torch


############################################
#    Noise Models
############################################


def createHistogram(bins, minVal, maxVal, observation, signal):
    """
    Creates a 2D histogram from 'observation' and 'signal'

    Parameters
    ----------
    bins: int
        The number of bins in x and y. The total number of 2D bins is 'bins'**2.
    minVal: float
        the lower bound of the lowest bin in x and y.
    maxVal: float
        the highest bound of the highest bin in x and y.
    observation: numpy array
        A 3D numpy array that is interpretted as a stack of 2D images.
        The number of images has to be divisible by the number of images in 'signal'.
        It is assumed that n subsequent images in observation belong to one image image in 'signal'.
    signal: numpy array
        A 3D numpy array that is interpretted as a stack of 2D images.

    Returns
    ----------
    histogram: numpy array
        A 3D array:
        'histogram[0,...]' holds the normalized 2D counts.
        Each row sums to 1, describing p(x_i|s_i).
        'histogram[1,...]' holds the lower boundaries of each bin in y.
        'histogram[2,...]' holds the upper boundaries of each bin in y.
        The values for x can be obtained by transposing 'histogram[1,...]' and 'histogram[2,...]'.
    """

    imgFactor = int(observation.shape[0] / signal.shape[0])
    histogram = np.zeros((3, bins, bins))
    ra = [minVal, maxVal]

    for i in range(observation.shape[0]):
        observation_ = observation[i].copy().ravel()

        signal_ = (signal[i // imgFactor].copy()).ravel()

        a = np.histogram2d(signal_, observation_, bins=bins, range=[ra, ra])
        histogram[0] = histogram[0] + a[0] + 1e-30  # This is for numerical stability

    for i in range(bins):
        if (
            np.sum(histogram[0, i, :]) > 1e-20
        ):  # We exclude empty rows from normalization
            histogram[0, i, :] /= np.sum(
                histogram[0, i, :]
            )  # we normalize each non-empty row

    for i in range(bins):
        histogram[1, :, i] = a[1][
            :-1
        ]  # The lower boundaries of each bin in y are stored in dimension 1
        histogram[2, :, i] = a[1][
            1:
        ]  # The upper boundaries of each bin in y are stored in dimension 2
        # The accordent numbers for x are just transopsed.

    return histogram


class NoiseModel:
    def __init__(self, histogram, device):
        """
        Creates a NoiseModel object.

        Parameters
        ----------
        histogram: numpy array
            A histogram as create by the 'createHistogram(...)' method.
        device:
            The device your NoiseModel lives on, e.g. your GPU.
        """
        self.device = device

        # The number of bins is the same in x and y
        bins = histogram.shape[1]

        # The lower boundaries of each bin in y are stored in dimension 1
        self.minv = np.min(histogram[1, ...])

        # The upper boundaries of each bin in y are stored in dimension 2
        self.maxv = np.max(histogram[2, ...])

        # move everything to GPU
        self.bins = torch.Tensor(np.array(float(bins))).to(self.device)
        self.fullHist = torch.Tensor(histogram[0, ...].astype(np.float32)).to(
            self.device
        )

    def likelihood(self, obs, signal):
        """
        Calculate the likelihood p(x_i|s_i) for every pixel in a tensor, using a histogram based noise model.
        To ensure differentiability in the direction of s_i, we linearly interpolate in this direction.

        Parameters
        ----------
        obs: pytorch tensor
            tensor holding your observed intesities x_i.

        signal: pytorch tensor
            tensor holding hypotheses for the clean signal at every pixel s_i^k.

        Returns
        ----------
        Torch tensor containing the observation likelihoods according to the noise model.
        """
        obsF = self.getIndexObsFloat(obs)
        obs_ = obsF.floor().long()
        signalF = self.getIndexSignalFloat(signal)
        signal_ = signalF.floor().long()
        fact = signalF - signal_.float()

        # Finally we are looking ud the values and interpolate
        return self.fullHist[signal_, obs_] * (1.0 - fact) + self.fullHist[
            torch.clamp((signal_ + 1).long(), 0, self.bins.long()), obs_
        ] * (fact)

    def getIndexObsFloat(self, x):
        return torch.clamp(
            self.bins * (x - self.minv) / (self.maxv - self.minv),
            min=0.0,
            max=self.bins - 1 - 1e-3,
        )

    def getIndexSignalFloat(self, x):
        return torch.clamp(
            self.bins * (x - self.minv) / (self.maxv - self.minv),
            min=0.0,
            max=self.bins - 1 - 1e-3,
        )
