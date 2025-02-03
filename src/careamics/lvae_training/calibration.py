from typing import Union, Optional

import numpy as np
import torch
from scipy import stats


def get_last_index(bin_count, quantile):
    cumsum = np.cumsum(bin_count)
    normalized_cumsum = cumsum / cumsum[-1]
    for i in range(1, len(normalized_cumsum)):
        if normalized_cumsum[-i] < quantile:
            return i - 1
    return None


def get_first_index(bin_count, quantile):
    cumsum = np.cumsum(bin_count)
    normalized_cumsum = cumsum / cumsum[-1]
    for i in range(len(normalized_cumsum)):
        if normalized_cumsum[i] > quantile:
            return i
    return None


class Calibration:
    """Calibrate the uncertainty computed over samples from LVAE model.

    Calibration is done by learning a scalar that maps the pixel-wise standard
    deviation of the the predicted samples into the actual prediction error.
    """

    def __init__(self, num_bins: int = 15):
        self._bins = num_bins
        self._bin_boundaries = None

    def compute_bin_boundaries(self, predict_std: np.ndarray) -> np.ndarray:
        """Compute the bin boundaries for `num_bins` bins and predicted std values."""
        min_std = np.min(predict_std)
        max_std = np.max(predict_std)
        return np.linspace(min_std, max_std, self._bins + 1)

    def compute_stats(
        self, pred: np.ndarray, pred_std: np.ndarray, target: np.ndarray
    ) -> dict[int, dict[str, Union[np.ndarray, list]]]:
        """
        It computes the bin-wise RMSE and RMV for each channel of the predicted image.

        Recall that:
            - RMSE = np.sqrt((pred - target)**2 / num_pixels)
            - RMV = np.sqrt(np.mean(pred_std**2))

        ALGORITHM
        - For each channel:
            - Given the bin boundaries, assign pixels of `std_ch` array to a specific bin index.
            - For each bin index:
                - Compute the RMSE, RMV, and number of pixels for that bin.

        NOTE: each channel of the predicted image/logvar has its own stats.

        Parameters
        ----------
        pred: np.ndarray
            Predicted patches, shape (n, h, w, c).
        pred_std: np.ndarray
            Std computed over the predicted patches, shape (n, h, w, c).
        target: np.ndarray
            Target GT image, shape (n, h, w, c).
        """
        self._bin_boundaries = {}
        stats_dict = {}
        for ch_idx in range(pred.shape[-1]):
            stats_dict[ch_idx] = {
                "bin_count": [],
                "rmv": [],
                "rmse": [],
                "bin_boundaries": None,
                "bin_matrix": [],
                "rmse_err": [],
            }
            pred_ch = pred[..., ch_idx]
            std_ch = pred_std[..., ch_idx]
            target_ch = target[..., ch_idx]
            boundaries = self.compute_bin_boundaries(std_ch)
            stats_dict[ch_idx]["bin_boundaries"] = boundaries
            bin_matrix = np.digitize(std_ch.reshape(-1), boundaries)
            bin_matrix = bin_matrix.reshape(std_ch.shape)
            stats_dict[ch_idx]["bin_matrix"] = bin_matrix
            error = (pred_ch - target_ch) ** 2
            for bin_idx in range(1, 1 + self._bins):
                bin_mask = bin_matrix == bin_idx
                bin_error = error[bin_mask]
                bin_size = np.sum(bin_mask)
                bin_error = (
                    np.sqrt(np.sum(bin_error) / bin_size) if bin_size > 0 else None
                )
                stderr = (
                    np.std(error[bin_mask]) / np.sqrt(bin_size)
                    if bin_size > 0
                    else None
                )
                rmse_stderr = np.sqrt(stderr) if stderr is not None else None

                bin_var = np.mean(std_ch[bin_mask] ** 2)
                stats_dict[ch_idx]["rmse"].append(bin_error)
                stats_dict[ch_idx]["rmse_err"].append(rmse_stderr)
                stats_dict[ch_idx]["rmv"].append(np.sqrt(bin_var))
                stats_dict[ch_idx]["bin_count"].append(bin_size)
        self.stats_dict = stats_dict
        return stats_dict

    def get_calibrated_factor_for_stdev(
        self,
        pred: Optional[np.ndarray] = None,
        pred_std: Optional[np.ndarray] = None,
        target: Optional[np.ndarray] = None,
        q_s: float = 0.00001,
        q_e: float = 0.99999,
    ) -> dict[str, float]:
        """Calibrate the uncertainty by multiplying the predicted std with a scalar.

        Parameters
        ----------
        stats_dict : dict[int, dict[str, Union[np.ndarray, list]]]
            Dictionary containing the stats for each channel.
        q_s : float, optional
            Start quantile, by default 0.00001.
        q_e : float, optional
            End quantile, by default 0.99999.

        Returns
        -------
        dict[str, float]
            Calibrated factor for each channel (slope + intercept).
        """
        if not hasattr(self, "stats_dict"):
            print("No stats found. Computing stats...")
            if any(v is None for v in [pred, pred_std, target]):
                raise ValueError("pred, pred_std, and target must be provided.")
            self.stats_dict = self.compute_stats(
                pred=pred, pred_std=pred_std, target=target
            )
        outputs = {}
        for ch_idx in self.stats_dict.keys():
            y = self.stats_dict[ch_idx]["rmse"]
            x = self.stats_dict[ch_idx]["rmv"]
            count = self.stats_dict[ch_idx]["bin_count"]

            first_idx = get_first_index(count, q_s)
            last_idx = get_last_index(count, q_e)
            x = x[first_idx:-last_idx]
            y = y[first_idx:-last_idx]
            slope, intercept, *_ = stats.linregress(x, y)
            output = {"scalar": slope, "offset": intercept}
            outputs[ch_idx] = output
        factors = self.get_factors_array(factors_dict=outputs)
        return outputs, factors

    def get_factors_array(self, factors_dict: list[dict]):
        """Get the calibration factors as a numpy array."""
        calib_scalar = [factors_dict[i]["scalar"] for i in range(len(factors_dict))]
        calib_scalar = np.array(calib_scalar).reshape(1, 1, 1, -1)
        calib_offset = [
            factors_dict[i].get("offset", 0.0) for i in range(len(factors_dict))
        ]
        calib_offset = np.array(calib_offset).reshape(1, 1, 1, -1)
        return {"scalar": calib_scalar, "offset": calib_offset}


def plot_calibration(ax, calibration_stats):
    first_idx = get_first_index(calibration_stats[0]["bin_count"], 0.0001)
    last_idx = get_last_index(calibration_stats[0]["bin_count"], 0.9999)
    ax.plot(
        calibration_stats[0]["rmv"][first_idx:-last_idx],
        calibration_stats[0]["rmse"][first_idx:-last_idx],
        "o",
        label=r"$\hat{C}_0$: Ch1",
    )

    first_idx = get_first_index(calibration_stats[1]["bin_count"], 0.0001)
    last_idx = get_last_index(calibration_stats[1]["bin_count"], 0.9999)
    ax.plot(
        calibration_stats[1]["rmv"][first_idx:-last_idx],
        calibration_stats[1]["rmse"][first_idx:-last_idx],
        "o",
        label=r"$\hat{C}_1$: Ch2",
    )
    # TODO add multichannel
    ax.set_xlabel("RMV")
    ax.set_ylabel("RMSE")
    ax.legend()
