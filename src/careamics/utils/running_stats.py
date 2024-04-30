"""Running stats submodule, used in the Zarr dataset."""

# from multiprocessing import Value
# from typing import Tuple

# import numpy as np


# class RunningStats:
#     """Calculates running mean and std."""

#     def __init__(self) -> None:
#         self.reset()

#     def reset(self) -> None:
#         """Reset the running stats."""
#         self.avg_mean = Value("d", 0)
#         self.avg_std = Value("d", 0)
#         self.m2 = Value("d", 0)
#         self.count = Value("i", 0)

#     def init(self, mean: float, std: float) -> None:
#         """Initialize running stats."""
#         with self.avg_mean.get_lock():
#             self.avg_mean.value += mean
#         with self.avg_std.get_lock():
#             self.avg_std.value = std

#     def compute_std(self) -> Tuple[float, float]:
#         """Compute std."""
#         if self.count.value >= 2:
#             self.avg_std.value = np.sqrt(self.m2.value / self.count.value)

#     def update(self, value: float) -> None:
#         """Update running stats."""
#         with self.count.get_lock():
#             self.count.value += 1
#         delta = value - self.avg_mean.value
#         with self.avg_mean.get_lock():
#             self.avg_mean.value += delta / self.count.value
#         delta2 = value - self.avg_mean.value
#         with self.m2.get_lock():
#             self.m2.value += delta * delta2
