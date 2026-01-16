import numpy as np
from numpy.typing import NDArray


class QuantileEstimator:
    """Streaming quantile estimator using adaptive histogram binning.

    Parameters
    ----------
    lower_quantiles : list[float]
        Lower quantile values to compute, one per channel.
    upper_quantiles : list[float]
        Upper quantile values to compute, one per channel.
    n_bins : int, optional
        Number of histogram bins. Default is 65536 (suitable for 16-bit data).
    margin : float, optional
        Fractional margin to add to histogram range. Default is 0.1 (10%).
    """

    def __init__(
        self,
        lower_quantiles: list[float],
        upper_quantiles: list[float],
        n_bins: int = 65536,
        margin: float = 0.1,
    ):
        self.n_bins = n_bins
        self.n_channels = len(lower_quantiles)
        self.margin = margin
        self.lower_quantiles = lower_quantiles
        self.upper_quantiles = upper_quantiles

        self.histograms = {
            ch: np.zeros(self.n_bins, dtype=np.int64) for ch in range(self.n_channels)
        }
        self.mins: dict[int, float] = dict.fromkeys(range(self.n_channels), np.inf)
        self.maxes: dict[int, float] = dict.fromkeys(range(self.n_channels), -np.inf)
        self.total_counts: dict[int, int] = dict.fromkeys(range(self.n_channels), 0)
        self.bin_edges: dict[int, NDArray] = {}

    def _rebin_histogram(
        self, old_hist: NDArray, old_edges: NDArray, new_edges: NDArray
    ) -> NDArray:
        new_hist = np.zeros(len(new_edges) - 1, dtype=np.int64)
        old_total = np.sum(old_hist)

        for i in range(len(old_hist)):
            if old_hist[i] == 0:
                continue

            old_bin_start = old_edges[i]
            old_bin_end = old_edges[i + 1]
            old_bin_width = old_bin_end - old_bin_start

            if old_bin_width == 0:
                bin_idx = np.searchsorted(new_edges, old_bin_start) - 1
                bin_idx = max(0, min(bin_idx, len(new_hist) - 1))
                new_hist[bin_idx] += old_hist[i]
                continue

            start_idx = max(
                0, np.searchsorted(new_edges, old_bin_start, side="right") - 1
            )
            end_idx = min(
                np.searchsorted(new_edges, old_bin_end, side="left"), len(new_hist)
            )

            fractional_counts = []
            target_indices = []
            for j in range(start_idx, end_idx):
                overlap_start = max(old_bin_start, new_edges[j])
                overlap_end = min(old_bin_end, new_edges[j + 1])
                overlap_width = max(0, overlap_end - overlap_start)
                proportion = overlap_width / old_bin_width
                fractional_counts.append(old_hist[i] * proportion)
                target_indices.append(j)

            if fractional_counts:
                int_counts = [int(c) for c in fractional_counts]
                remainders = [c - int(c) for c in fractional_counts]
                total_distributed = sum(int_counts)
                counts_to_add = old_hist[i] - total_distributed

                if counts_to_add > 0:
                    sorted_indices = sorted(
                        range(len(remainders)),
                        key=lambda k: remainders[k],
                        reverse=True,
                    )
                    for k in range(min(counts_to_add, len(sorted_indices))):
                        int_counts[sorted_indices[k]] += 1

                for j, count in zip(target_indices, int_counts, strict=True):
                    new_hist[j] += count

        new_total = np.sum(new_hist)
        if new_total != old_total and old_total > 0:
            diff = old_total - new_total
            if diff != 0:
                new_hist[np.argmax(new_hist)] += diff

        return new_hist

    def update(self, patch: NDArray) -> None:
        flattened_patch = patch.reshape(self.n_channels, -1)

        for ch in range(self.n_channels):
            channel_data = flattened_patch[ch]
            current_min = float(np.min(channel_data))
            current_max = float(np.max(channel_data))

            self.mins[ch] = min(self.mins[ch], current_min)
            self.maxes[ch] = max(self.maxes[ch], current_max)
            self.total_counts[ch] += len(channel_data)

            current_range = self.maxes[ch] - self.mins[ch]
            if current_range == 0:
                current_range = 1.0

            if ch not in self.bin_edges:
                self.bin_edges[ch] = np.linspace(
                    self.mins[ch] - self.margin * current_range,
                    self.maxes[ch] + self.margin * current_range,
                    self.n_bins + 1,
                )
            else:
                bin_min = self.bin_edges[ch][0]
                bin_max = self.bin_edges[ch][-1]

                if self.mins[ch] < bin_min or self.maxes[ch] > bin_max:
                    expanded_min = min(self.mins[ch], bin_min)
                    expanded_max = max(self.maxes[ch], bin_max)
                    expanded_range = expanded_max - expanded_min
                    if expanded_range == 0:
                        expanded_range = max(bin_max - bin_min, 1.0)

                    new_edges = np.linspace(
                        expanded_min - self.margin * expanded_range,
                        expanded_max + self.margin * expanded_range,
                        self.n_bins + 1,
                    )
                    self.histograms[ch] = self._rebin_histogram(
                        self.histograms[ch], self.bin_edges[ch], new_edges
                    )
                    self.bin_edges[ch] = new_edges

            hist, _ = np.histogram(channel_data, bins=self.bin_edges[ch], density=False)
            self.histograms[ch] += hist

    def _calculate_quantile(self, ch: int, quantile: float) -> float:
        if self.total_counts[ch] == 0:
            return 0.0

        if quantile == 0:
            return float(self.mins[ch])
        if quantile == 1:
            return float(self.maxes[ch])

        cumsum = np.cumsum(self.histograms[ch])
        target_count = self.total_counts[ch] * quantile

        bin_idx = int(np.searchsorted(cumsum, target_count))
        bin_idx = max(0, min(bin_idx, len(self.histograms[ch]) - 1))

        count_in_bin = self.histograms[ch][bin_idx]
        if count_in_bin == 0:
            return float(self.bin_edges[ch][bin_idx])

        counts_before = cumsum[bin_idx - 1] if bin_idx > 0 else 0
        counts_needed = max(0, min(target_count - counts_before, count_in_bin))
        fraction = counts_needed / count_in_bin

        bin_start = self.bin_edges[ch][bin_idx]
        bin_end = self.bin_edges[ch][bin_idx + 1]
        return float(bin_start + fraction * (bin_end - bin_start))

    def finalize(self) -> tuple[NDArray, NDArray]:
        lower = np.array(
            [
                self._calculate_quantile(ch, self.lower_quantiles[ch])
                for ch in range(self.n_channels)
            ]
        )
        upper = np.array(
            [
                self._calculate_quantile(ch, self.upper_quantiles[ch])
                for ch in range(self.n_channels)
            ]
        )
        return lower, upper
