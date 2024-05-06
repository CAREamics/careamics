"""
Script for utility functions needed by the LVAE model.
"""

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