"""Datasets for the COSDD reproduction runs.

Each dataset gives the noisy stack used for training (S, Y, X), and the test
stack with its ground truth, following the protocol of the COSDD paper
(Salmon & Krull, WACV 2025, https://arxiv.org/abs/2310.07887).

- `conva`: Convallaria A (structured Convallaria, EMCCD, noise correlated along
  x). 100 frames of 1024 x 1024. Protocol of the WACV_reproducibility branch of
  krulllab/COSDD: train on all frames (90/10 split by frame), test on every
  frame cropped to [:512, :512], ground truth = mean of the 100 noisy crops.

- `actin`: Actin Confocal (Hagen et al. 2021, LSCM, noise correlated along x),
  the dataset of the COSDD demo notebooks. 79 frames of 1024 x 1024. The paper
  gives no split, so we train and test on all frames. The high-SNR reference is
  on another intensity scale (about 5.7x brighter), so, as in prediction.ipynb
  of the DL4MIA branch of krulllab/COSDD (Hagen mito-confocal), it is mapped
  onto the noisy scale by least squares, gt = a * high_snr + b, fitted over all
  test frames.

"""

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from numpy.typing import NDArray

DATA_DIR = Path("/group/jug/Igor/cosdd/data")

CONVA_PATH = Path("/group/jug/public_html/n2v/flower.tif")

ACTIN_URL = (
    "https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/"
    "10.5524/100001_101000/100888/03-actin-confocal/actin-confocal-{snr}.tif"
)

# PSNR of "Ours large" in the paper, MMSE over 1 / 10 / 100 samples
# (main paper Table 1 and supplementary Table 1).
PAPER_PSNR = {
    "conva": {"sample": 36.54, "mmse10": 37.39, "mmse100": 37.49},
    "actin": {"sample": 26.81, "mmse10": 27.37, "mmse100": 27.42},
}


@dataclass
class COSDDData:
    """Noisy training stack and test stack with ground truth."""

    name: str
    noise_direction: str
    train: NDArray  # (S, Y, X), float32, all noisy frames used for training
    test: NDArray  # (S, Y, X), float32, noisy test frames
    gt: NDArray  # (S, Y, X), float32, ground truth of the test frames


def _download(url: str, path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {url} -> {path}")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(path)
    return path


def load_conva() -> COSDDData:
    noisy = tifffile.imread(CONVA_PATH).astype(np.float32)
    test = noisy[:, :512, :512]
    gt = np.repeat(test.mean(axis=0, keepdims=True), len(test), axis=0)
    return COSDDData("conva", "x", train=noisy, test=test, gt=gt)


def load_actin() -> COSDDData:
    low = _download(
        ACTIN_URL.format(snr="lowsnr"), DATA_DIR / "actin-confocal-lowsnr.tif"
    )
    high = _download(
        ACTIN_URL.format(snr="highsnr"), DATA_DIR / "actin-confocal-highsnr.tif"
    )
    noisy = tifffile.imread(low).astype(np.float32)
    gt = match_scale(tifffile.imread(high).astype(np.float32), noisy)
    return COSDDData("actin", "x", train=noisy, test=noisy, gt=gt)


def match_scale(reference: NDArray, noisy: NDArray) -> NDArray:
    """Map `reference` onto the scale of `noisy`, a * reference + b, least squares.

    Same as `utils.minimise_mse` of the DL4MIA branch of krulllab/COSDD: the noisy
    images are regressed on the reference over all frames jointly.
    """
    x = reference.astype(np.float64).ravel()
    y = noisy.astype(np.float64).ravel()
    a = np.cov(x, y, bias=True)[0, 1] / x.var()
    b = y.mean() - a * x.mean()
    return (a * reference + b).astype(np.float32)


LOADERS = {"conva": load_conva, "actin": load_actin}


def load(name: str) -> COSDDData:
    return LOADERS[name]()


def train_val_split(
    n_frames: int, train_split: float = 0.9, seed: int = 42
) -> tuple[NDArray, NDArray]:
    """Split frame indices into training and validation, as the reference does."""
    perm = np.random.default_rng(seed).permutation(n_frames)
    n_train = int(n_frames * train_split)
    return np.sort(perm[:n_train]), np.sort(perm[n_train:])
