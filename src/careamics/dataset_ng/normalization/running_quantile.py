import numpy as np
from numpy.typing import NDArray


class QuantileEstimator:
    def __init__(
        self,
        lower_quantiles: list[float],
        upper_quantiles: list[float],
        n_bins: int = 10000,
        margin: float = 0.01,
    ):
        self.n_bins = n_bins
        self.n_channels = len(lower_quantiles)
        self.histograms = {
            ch: np.zeros(self.n_bins, dtype=np.int64) for ch in range(self.n_channels)
        }
        self.mins = dict.fromkeys(range(self.n_channels), np.inf)
        self.maxes = dict.fromkeys(range(self.n_channels), -np.inf)
        self.total_counts = dict.fromkeys(range(self.n_channels), 0)
        self.bin_edges = {}
        self.lower_quantiles = lower_quantiles
        self.upper_quantiles = upper_quantiles
        self.margin = margin

    def _rebin_histogram(
        self, old_hist: NDArray, old_edges: NDArray, new_edges: NDArray
    ) -> NDArray:
        new_hist = np.zeros(len(new_edges) - 1, dtype=np.int64)

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

            # Find which new bins overlap with this old bin
            # Find first new bin that starts >= old_bin_start
            start_idx = np.searchsorted(new_edges, old_bin_start, side="right") - 1
            start_idx = max(0, start_idx)

            # Find last new bin that ends <= old_bin_end
            end_idx = np.searchsorted(new_edges, old_bin_end, side="left")
            end_idx = min(end_idx, len(new_hist))

            # Distribute counts proportionally
            for j in range(start_idx, end_idx):
                new_bin_start = new_edges[j]
                new_bin_end = new_edges[j + 1]

                # Calculate overlap
                overlap_start = max(old_bin_start, new_bin_start)
                overlap_end = min(old_bin_end, new_bin_end)
                overlap_width = max(0, overlap_end - overlap_start)

                # Proportion of old bin that overlaps with this new bin
                proportion = overlap_width / old_bin_width
                new_hist[j] += int(old_hist[i] * proportion)

        return new_hist

    def update(self, patch: NDArray):
        flattened_patch = patch.reshape(self.n_channels, -1)

        for ch in range(self.n_channels):
            channel_data = flattened_patch[ch]
            current_min = np.min(channel_data)
            current_max = np.max(channel_data)

            self.mins[ch] = min(self.mins[ch], current_min)
            self.maxes[ch] = max(self.maxes[ch], current_max)

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
                    # Create new bin edges with expanded range
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
            self.total_counts[ch] += len(channel_data)

    def calculate_quantile(self, quantiles: list[float]) -> NDArray:
        result = []

        for ch in range(self.n_channels):
            if self.total_counts[ch] == 0:
                result.append(0.0)
                continue

            cumsum = np.cumsum(self.histograms[ch])
            target_count = self.total_counts[ch] * quantiles[ch]
            
            # Handle edge case: if target_count is 0 or negative, return actual min
            if target_count <= 0:
                result.append(float(self.mins[ch]))
                continue
            
            # Handle edge case: if target_count >= total_counts, return actual max
            if target_count >= self.total_counts[ch]:
                result.append(float(self.maxes[ch]))
                continue

            # Find the bin containing the target quantile
            # np.searchsorted returns the first index where cumsum[i] >= target_count
            bin_idx = int(np.searchsorted(cumsum, target_count))
            
            # Clamp to valid range
            bin_idx = min(bin_idx, len(self.histograms[ch]) - 1)
            bin_idx = max(bin_idx, 0)

            count_in_bin = self.histograms[ch][bin_idx]
            
            if count_in_bin == 0:
                # Empty bin - return the bin start
                result.append(float(self.bin_edges[ch][bin_idx]))
            else:
                # Calculate how many counts we need from this bin
                counts_before = cumsum[bin_idx - 1] if bin_idx > 0 else 0
                counts_needed = target_count - counts_before
                
                # Clamp counts_needed to valid range [0, count_in_bin]
                counts_needed = max(0, min(counts_needed, count_in_bin))
                
                # Interpolate within the bin
                fraction = counts_needed / count_in_bin
                bin_start = self.bin_edges[ch][bin_idx]
                bin_end = self.bin_edges[ch][bin_idx + 1]
                
                result.append(float(bin_start + fraction * (bin_end - bin_start)))

        return np.array(result)

    def finalize(self) -> tuple[NDArray, NDArray]:
        lower_quantiles = self.calculate_quantile(self.lower_quantiles)
        upper_quantiles = self.calculate_quantile(self.upper_quantiles)
        return lower_quantiles, upper_quantiles
