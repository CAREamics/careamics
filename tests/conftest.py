from pathlib import Path
from typing import Callable

import pytest
import numpy as np


def create_tiff(path: Path, n_files: int):
    """Create tiff files for testing."""
    if not path.exists():
        path.mkdir()

    for i in range(n_files):
        file_path = path / f"file_{i}.tif"
        file_path.touch()


@pytest.fixture
def test_config(tmp_path) -> dict:
    # create data
    ext = "tif"

    path_train = tmp_path / "train"
    create_tiff(path_train, n_files=3)

    path_validation = tmp_path / "validation"
    create_tiff(path_validation, n_files=1)

    path_test = tmp_path / "test"
    create_tiff(path_test, n_files=2)

    # create dictionary
    test_configuration = {
        "experiment_name": "testing",
        "workdir": str(tmp_path),
        "algorithm": {
            "loss": ["n2v"],
            "model": "UNET",
            "num_masked_pixels": 0.2,
            "pixel_manipulation": "n2v",
        },
        "training": {
            "num_epochs": 100,
            "learning_rate": 0.0001,
            "optimizer": {
                "name": "Adam",
                "parameters": {
                    "lr": 0.001,
                    "betas": [0.9, 0.999],
                    "eps": 1e-08,
                    "weight_decay": 0.0005,
                    "amsgrad": True,
                },
            },
            "lr_scheduler": {
                "name": "ReduceLROnPlateau",
                "parameters": {"factor": 0.5, "patience": 5, "mode": "min"},
            },
            "amp": {
                "toggle": False,
                "init_scale": 1024,
            },
            "data": {
                "path": str(path_train),
                "ext": ext,
                "axes": "YX",
                "extraction_strategy": "sequential",
                "patch_size": [128, 128],
                "batch_size": 1,
            },
        },
        "evaluation": {
            "data": {
                "path": str(path_validation),
                "ext": ext,
                "axes": "YX",
                "extraction_strategy": "sequential",
                "patch_size": [128, 128],
                "batch_size": 1,
            },
            "metric": "psnr",
        },
        "prediction": {
            "data": {
                "path": str(path_test),
                "ext": ext,
                "axes": "YX",
                "extraction_strategy": "sequential",
                "patch_size": [128, 128],
                "batch_size": 1,
            },
            "overlap": [25, 25],
        },
    }

    return test_configuration


@pytest.fixture
def ordered_array() -> Callable:
    """A function that returns an array with ordered values."""

    def _ordered_array(shape: tuple) -> np.ndarray:
        """An array with ordered values.

        Parameters
        ----------
        shape : tuple
            Shape of the array.

        Returns
        -------
        np.ndarray
            Array with ordered values.
        """
        return np.arange(np.prod(shape)).reshape(shape)

    return _ordered_array


@pytest.fixture
def array_2D() -> np.ndarray:
    """A 2D array with shape (1, 10, 9).

    Returns
    -------
    np.ndarray
        2D array with shape (1, 10, 9).
    """
    return np.array(
        [
            [
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
                [31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
                [41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
                [51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
                [61, 62, 63, 64, 65, 66, 67, 68, 69, 70],
                [71, 72, 73, 74, 75, 76, 77, 78, 79, 80],
                [81, 82, 83, 84, 85, 86, 87, 88, 89, 90],
            ]
        ]
    )


@pytest.fixture
def array_3D() -> np.ndarray:
    """A 3D array with shape (1, 5, 10, 9).

    Returns
    -------
    np.ndarray
        3D array with shape (1, 5, 10, 9).
    """
    return np.array(
        [
            [
                [
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                    [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
                    [31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
                    [41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
                    [51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
                    [61, 62, 63, 64, 65, 66, 67, 68, 69, 70],
                    [71, 72, 73, 74, 75, 76, 77, 78, 79, 80],
                    [81, 82, 83, 84, 85, 86, 87, 88, 89, 90],
                ],
                [
                    [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
                    [111, 112, 113, 114, 115, 116, 117, 118, 119, 120],
                    [121, 122, 123, 124, 125, 126, 127, 128, 129, 130],
                    [131, 132, 133, 134, 135, 136, 137, 138, 139, 140],
                    [141, 142, 143, 144, 145, 146, 147, 148, 149, 150],
                    [151, 152, 153, 154, 155, 156, 157, 158, 159, 160],
                    [161, 162, 163, 164, 165, 166, 167, 168, 169, 170],
                    [171, 172, 173, 174, 175, 176, 177, 178, 179, 180],
                    [181, 182, 183, 184, 185, 186, 187, 188, 189, 190],
                ],
                [
                    [201, 202, 203, 204, 205, 206, 207, 208, 209, 210],
                    [211, 212, 213, 214, 215, 216, 217, 218, 219, 220],
                    [221, 222, 223, 224, 225, 226, 227, 228, 229, 230],
                    [231, 232, 233, 234, 235, 236, 237, 238, 239, 240],
                    [241, 242, 243, 244, 245, 246, 247, 248, 249, 250],
                    [251, 252, 253, 254, 255, 256, 257, 258, 259, 260],
                    [261, 262, 263, 264, 265, 266, 267, 268, 269, 270],
                    [271, 272, 273, 274, 275, 276, 277, 278, 279, 280],
                    [281, 282, 283, 284, 285, 286, 287, 288, 289, 290],
                ],
                [
                    [301, 302, 303, 304, 305, 306, 307, 308, 309, 310],
                    [311, 312, 313, 314, 315, 316, 317, 318, 319, 320],
                    [321, 322, 323, 324, 325, 326, 327, 328, 329, 330],
                    [331, 332, 333, 334, 335, 336, 337, 338, 339, 340],
                    [341, 342, 343, 344, 345, 346, 347, 348, 349, 350],
                    [351, 352, 353, 354, 355, 356, 357, 358, 359, 360],
                    [361, 362, 363, 364, 365, 366, 367, 368, 369, 370],
                    [371, 372, 373, 374, 375, 376, 377, 378, 379, 380],
                    [381, 382, 383, 384, 385, 386, 387, 388, 389, 390],
                ],
                [
                    [401, 402, 403, 404, 405, 406, 407, 408, 409, 410],
                    [411, 412, 413, 414, 415, 416, 417, 418, 419, 420],
                    [421, 422, 423, 424, 425, 426, 427, 428, 429, 430],
                    [431, 432, 433, 434, 435, 436, 437, 438, 439, 440],
                    [441, 442, 443, 444, 445, 446, 447, 448, 449, 450],
                    [451, 452, 453, 454, 455, 456, 457, 458, 459, 460],
                    [461, 462, 463, 464, 465, 466, 467, 468, 469, 470],
                    [471, 472, 473, 474, 475, 476, 477, 478, 479, 480],
                    [481, 482, 483, 484, 485, 486, 487, 488, 489, 490],
                ],
            ]
        ]
    )
