"""
Script for utility functions needed by the model
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