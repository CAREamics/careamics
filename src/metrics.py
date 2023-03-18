import numpy as np


def psnr_base(gt, pred, range_=255.0):
    mse = np.mean((gt - pred)**2)
    # print(gt.max())
    # print(pred.max())
    # print('MSE', mse, np.log10(range_))
    # print(20 * np.log10((range_)/np.sqrt(mse)), 10 * np.log10((range_ ** 2) / mse))
    return 20 * np.log10((range_)/np.sqrt(mse))


def zero_mean(x):
    return x-np.mean(x)


def fix_range(gt,x):
    a = np.sum(gt*x) / (np.sum(x*x))
    return x*a


def fix(gt,x):
    gt_ = zero_mean(gt)
    return fix_range(gt_, zero_mean(x))


def scale_invariant_psnr(gt, pred):
    range = (np.max(gt) - np.min(gt)) / np.std(gt)
    gt_ = zero_mean(gt) / np.std(gt)
    return psnr_base(zero_mean(gt_), fix(gt_, pred), range)


class MetricTracker:

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value: int, n: int = 1):
        """Update the metric tracker state

        Parameters
        ----------
        value : int
            Value to update the metric tracker with
        n : int
            Number of values, equals to batch size
        """
        self.val = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count