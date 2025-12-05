import numpy as np
from numpy.typing import NDArray


class QuantileEstimator:
    """Streaming quantile estimator using adaptive histogram binning.

    Uses a hybrid approach: stores exact values for small datasets (providing
    exact quantile computation), then switches to histogram-based estimation
    for larger datasets where exact computation would be memory-prohibitive.

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
    exact_threshold : int, optional
        Maximum number of values per channel to store for exact computation.
        When exceeded, switches to histogram mode. Default is 100000.
    """

    def __init__(
        self,
        lower_quantiles: list[float],
        upper_quantiles: list[float],
        n_bins: int = 65536,
        margin: float = 0.1,
        exact_threshold: int = 100_000,
    ):
        self.n_bins = n_bins
        self.n_channels = len(lower_quantiles)
        self.exact_threshold = exact_threshold

        # Exact storage for small datasets
        self._exact_values: dict[int, list[NDArray]] = {
            ch: [] for ch in range(self.n_channels)
        }
        self._use_histogram: dict[int, bool] = dict.fromkeys(
            range(self.n_channels), False
        )

        # Histogram storage for large datasets
        self.histograms = {
            ch: np.zeros(self.n_bins, dtype=np.int64) for ch in range(self.n_channels)
        }
        self.mins = dict.fromkeys(range(self.n_channels), np.inf)
        self.maxes = dict.fromkeys(range(self.n_channels), -np.inf)
        self.total_counts = dict.fromkeys(range(self.n_channels), 0)
        self.bin_edges: dict[int, NDArray] = {}
        self.lower_quantiles = lower_quantiles
        self.upper_quantiles = upper_quantiles
        self.margin = margin

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

            start_idx = np.searchsorted(new_edges, old_bin_start, side="right") - 1
            start_idx = max(0, start_idx)

            end_idx = np.searchsorted(new_edges, old_bin_end, side="left")
            end_idx = min(end_idx, len(new_hist))

            # Calculate fractional counts for all target bins
            fractional_counts = []
            target_indices = []
            for j in range(start_idx, end_idx):
                new_bin_start = new_edges[j]
                new_bin_end = new_edges[j + 1]

                overlap_start = max(old_bin_start, new_bin_start)
                overlap_end = min(old_bin_end, new_bin_end)
                overlap_width = max(0, overlap_end - overlap_start)

                proportion = overlap_width / old_bin_width
                fractional_counts.append(old_hist[i] * proportion)
                target_indices.append(j)

            # Distribute counts preserving total using largest remainder method
            if fractional_counts:
                int_counts = [int(c) for c in fractional_counts]
                remainders = [c - int(c) for c in fractional_counts]
                total_distributed = sum(int_counts)
                counts_to_add = old_hist[i] - total_distributed

                # Add remaining counts to bins with largest remainders
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

        # Final verification and correction if needed
        new_total = np.sum(new_hist)
        if new_total != old_total and old_total > 0:
            # Distribute any remaining difference to the bin with most counts
            diff = old_total - new_total
            if diff != 0:
                max_bin = np.argmax(new_hist)
                new_hist[max_bin] += diff

        return new_hist

    def _convert_to_histogram(self, ch: int) -> None:
        all_values = np.concatenate(self._exact_values[ch])
        self._exact_values[ch] = []

        current_range = self.maxes[ch] - self.mins[ch]
        if current_range == 0:
            current_range = 1.0

        self.bin_edges[ch] = np.linspace(
            self.mins[ch] - self.margin * current_range,
            self.maxes[ch] + self.margin * current_range,
            self.n_bins + 1,
        )

        hist, _ = np.histogram(all_values, bins=self.bin_edges[ch], density=False)
        self.histograms[ch] = hist
        self._use_histogram[ch] = True

    def update(self, patch: NDArray) -> None:
        flattened_patch = patch.reshape(self.n_channels, -1)

        for ch in range(self.n_channels):
            channel_data = flattened_patch[ch]
            current_min = np.min(channel_data)
            current_max = np.max(channel_data)

            self.mins[ch] = min(self.mins[ch], current_min)
            self.maxes[ch] = max(self.maxes[ch], current_max)
            self.total_counts[ch] += len(channel_data)

            # Check if we should switch to histogram mode
            if not self._use_histogram[ch]:
                self._exact_values[ch].append(channel_data.copy())

                if self.total_counts[ch] > self.exact_threshold:
                    self._convert_to_histogram(ch)
                continue

            # Histogram mode
            current_range = self.maxes[ch] - self.mins[ch]

            if ch not in self.bin_edges:
                if current_range == 0:
                    current_range = 1.0
                self.bin_edges[ch] = np.linspace(
                    self.mins[ch] - self.margin * current_range,
                    self.maxes[ch] + self.margin * current_range,
                    self.n_bins + 1,
                )
            else:
                bin_min = self.bin_edges[ch][0]
                bin_max = self.bin_edges[ch][-1]
                needs_rebin = self.mins[ch] < bin_min or self.maxes[ch] > bin_max

                if needs_rebin:
                    expanded_min = min(self.mins[ch], bin_min)
                    expanded_max = max(self.maxes[ch], bin_max)
                    expanded_range = expanded_max - expanded_min

                    if expanded_range == 0:
                        expanded_range = max(bin_max - bin_min, 1.0)

                    new_min = expanded_min - self.margin * expanded_range
                    new_max = expanded_max + self.margin * expanded_range
                    new_edges = np.linspace(new_min, new_max, self.n_bins + 1)

                    self.histograms[ch] = self._rebin_histogram(
                        self.histograms[ch], self.bin_edges[ch], new_edges
                    )
                    self.bin_edges[ch] = new_edges

            hist, _ = np.histogram(channel_data, bins=self.bin_edges[ch], density=False)
            self.histograms[ch] += hist

    def _calculate_quantile_exact(self, ch: int, quantile: float) -> float:
        """Calculate quantile using exact stored values."""
        if not self._exact_values[ch]:
            return 0.0
        all_values = np.concatenate(self._exact_values[ch])
        return float(np.quantile(all_values, quantile))

    def _calculate_quantile_histogram(self, ch: int, quantile: float) -> float:
        """Calculate quantile using histogram approximation."""
        if self.total_counts[ch] == 0:
            return 0.0

        cumsum = np.cumsum(self.histograms[ch])
        target_count = self.total_counts[ch] * quantile

        if target_count <= 0:
            return float(self.mins[ch])

        if target_count >= self.total_counts[ch]:
            return float(self.maxes[ch])

        bin_idx = int(np.searchsorted(cumsum, target_count))
        bin_idx = min(bin_idx, len(self.histograms[ch]) - 1)
        bin_idx = max(bin_idx, 0)

        count_in_bin = self.histograms[ch][bin_idx]

        if count_in_bin == 0:
            return float(self.bin_edges[ch][bin_idx])

        counts_before = cumsum[bin_idx - 1] if bin_idx > 0 else 0
        counts_needed = target_count - counts_before
        counts_needed = max(0, min(counts_needed, count_in_bin))

        fraction = counts_needed / count_in_bin
        bin_start = self.bin_edges[ch][bin_idx]
        bin_end = self.bin_edges[ch][bin_idx + 1]

        return float(bin_start + fraction * (bin_end - bin_start))

    def calculate_quantile(self, quantiles: list[float]) -> NDArray:
        """Calculate quantiles for all channels.

        Parameters
        ----------
        quantiles : list[float]
            Quantile values to compute, one per channel.

        Returns
        -------
        NDArray
            Array of computed quantile values, one per channel.
        """
        result = []

        for ch in range(self.n_channels):
            if self._use_histogram[ch]:
                result.append(self._calculate_quantile_histogram(ch, quantiles[ch]))
            else:
                result.append(self._calculate_quantile_exact(ch, quantiles[ch]))

        return np.array(result)

    def finalize(self) -> tuple[NDArray, NDArray]:
        lower_quantiles = self.calculate_quantile(self.lower_quantiles)
        upper_quantiles = self.calculate_quantile(self.upper_quantiles)
        return lower_quantiles, upper_quantiles
