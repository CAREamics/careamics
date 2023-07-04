import numpy as np


def normalize(img: np.ndarray, mean: float, std: float) -> np.ndarray:
    zero_mean = img - mean
    return zero_mean / std


def denormalize(img: np.ndarray, mean: float, std: float) -> np.ndarray:
    return img * std + mean
