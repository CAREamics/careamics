"""
Script for utility functions needed by the LVAE model.
"""
import torch
import torch.nn as nn 

def torch_nanmean(inp):
    return torch.mean(inp[~inp.isnan()])

def compute_batch_mean(x):
    N = len(x)
    return x.view(N, -1).mean(dim=1)

def normalize_input(self, x):
    if self.normalized_input:
        return x
    return (x - self.data_mean['input'].mean()) / self.data_std['input'].mean()

def normalize_target(self, target, batch=None):
    return (target - self.data_mean['target']) / self.data_std['target']

def unnormalize_target(self, target_normalized):
    return target_normalized * self.data_std['target'] + self.data_mean['target']

def power_of_2(self, x):
    assert isinstance(x, int)
    if x == 1:
        return True
    if x == 0:
        # happens with validation
        return False
    if x % 2 == 1:
        return False
    return self.power_of_2(x // 2)

class Enum:
    @classmethod
    def name(cls, enum_type):
        for key, value in cls.__dict__.items():
            if enum_type == value:
                return key

    @classmethod
    def contains(cls, enum_type):
        for key, value in cls.__dict__.items():
            if enum_type == value:
                return True
        return False

    @classmethod
    def from_name(cls, enum_type_str):
        for key, value in cls.__dict__.items():
            if key == enum_type_str:
                return value
        assert f'{cls.__name__}:{enum_type_str} doesnot exist.'

class LossType(Enum):
    Elbo = 0
    ElboWithCritic = 1
    ElboMixedReconstruction = 2
    MSE = 3
    ElboWithNbrConsistency = 4
    ElboSemiSupMixedReconstruction = 5
    ElboCL = 6
    ElboRestrictedReconstruction = 7
    DenoiSplitMuSplit = 8


def _pad_crop_img(x, size, mode) -> torch.Tensor:
    """ Pads or crops a tensor.
    Pads or crops a tensor of shape (batch, channels, h, w) to new height
    and width given by a tuple.
    Args:
        x (torch.Tensor): Input image
        size (list or tuple): Desired size (height, width)
        mode (str): Mode, either 'pad' or 'crop'
    Returns:
        The padded or cropped tensor
    """

    assert x.dim() == 4 and len(size) == 2
    size = tuple(size)
    x_size = x.size()[2:4]
    if mode == 'pad':
        cond = x_size[0] > size[0] or x_size[1] > size[1]
    elif mode == 'crop':
        cond = x_size[0] < size[0] or x_size[1] < size[1]
    else:
        raise ValueError("invalid mode '{}'".format(mode))
    if cond:
        raise ValueError('trying to {} from size {} to size {}'.format(mode, x_size, size))
    dr, dc = (abs(x_size[0] - size[0]), abs(x_size[1] - size[1]))
    dr1, dr2 = dr // 2, dr - (dr // 2)
    dc1, dc2 = dc // 2, dc - (dc // 2)
    if mode == 'pad':
        return nn.functional.pad(x, [dc1, dc2, dr1, dr2, 0, 0, 0, 0])
    elif mode == 'crop':
        return x[:, :, dr1:x_size[0] - dr2, dc1:x_size[1] - dc2]

    
def pad_img_tensor(x, size) -> torch.Tensor:
    """Pads a tensor.
    Pads a tensor of shape (batch, channels, h, w) to a desired height and width.
    Args:
        x (torch.Tensor): Input image
        size (list or tuple): Desired size (height, width)
    Returns:
        The padded tensor
    """

    return _pad_crop_img(x, size, 'pad')


def crop_img_tensor(x, size) -> torch.Tensor:
    """Crops a tensor.
    Crops a tensor of shape (batch, channels, h, w) to a desired height and width
    given by a tuple.
    Args:
        x (torch.Tensor): Input image
        size (list or tuple): Desired size (height, width)
    Returns:
        The cropped tensor
    """
    return _pad_crop_img(x, size, 'crop')


class StableExponential:
    """
    Class that redefines the definition of exp() to increase numerical stability. 
    Naturally, also the definition of log() must change accordingly. 
    However, it is worth noting that the two operations remain one the inverse of 
    the other, meaning that x = log(exp(x)) and x = exp(log(x)) are always true.
    
    NOTE 1:
        Here, the idea is that everything is done on the tensor which you've given in the constructor.
        when exp() is called, what that means is that we want to compute self._tensor.exp()
        when log() is called, we want to compute torch.log(self._tensor.exp())
    
    NOTE 2:
        Given the output from exp(), torch.log() or the log() method of the class give identical results.
    """

    def __init__(self, tensor):
        self._raw_tensor = tensor
        posneg_dic = self.posneg_separation(self._raw_tensor)
        self.pos_f, self.neg_f = posneg_dic['filter']
        self.pos_data, self.neg_data = posneg_dic['value']

    def posneg_separation(self, tensor):
        pos = tensor > 0
        pos_tensor = torch.clip(tensor, min=0)

        neg = tensor <= 0
        neg_tensor = torch.clip(tensor, max=0)

        return {'filter': [pos, neg], 'value': [pos_tensor, neg_tensor]}

    def exp(self):
        return torch.exp(self.neg_data) * self.neg_f + (1 + self.pos_data) * self.pos_f

    def log(self):
        return self.neg_data * self.neg_f + torch.log(1 + self.pos_data) * self.pos_f


class StableLogVar:

    def __init__(self, logvar, enable_stable=True, var_eps=1e-6):
        """
        Args:
            var_eps: var() has this minimum value.
        """
        self._lv = logvar
        self._enable_stable = enable_stable
        self._eps = var_eps

    def get(self):
        if self._enable_stable is False:
            return self._lv

        return torch.log(self.get_var())

    def get_var(self):
        if self._enable_stable is False:
            return torch.exp(self._lv)
        return StableExponential(self._lv).exp() + self._eps

    def get_std(self):
        return torch.sqrt(self.get_var())

    def centercrop_to_size(self, size):
        if self._lv.shape[-1] == size:
            return

        diff = self._lv.shape[-1] - size
        assert diff > 0 and diff % 2 == 0
        self._lv = F.center_crop(self._lv, (size, size))


class StableMean:

    def __init__(self, mean):
        self._mean = mean

    def get(self):
        return self._mean

    def centercrop_to_size(self, size):
        if self._mean.shape[-1] == size:
            return

        diff = self._mean.shape[-1] - size
        assert diff > 0 and diff % 2 == 0
        self._mean = F.center_crop(self._mean, (size, size))
