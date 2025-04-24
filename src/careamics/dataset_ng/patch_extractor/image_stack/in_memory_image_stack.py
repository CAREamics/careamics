from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Union

import numpy as np
from numpy.typing import DTypeLike, NDArray
from typing_extensions import Self

from careamics.dataset.dataset_utils import reshape_array
from careamics.file_io.read import ReadFunc, read_tiff


class InMemoryImageStack:
    """
    A class for extracting patches from an image stack that has been loaded into memory.
    """

    def __init__(self, source: Union[Path, Literal["array"]], data: NDArray):
        self.source: Union[Path, Literal["array"]] = source
        # data expected to be in SC(Z)YX shape, reason to use from_array constructor
        self._data: NDArray = data
        self.data_shape: Sequence[int] = self._data.shape
        self.data_dtype: DTypeLike = self._data.dtype
        self.collocate_patch_region: bool = False  # TODO subclass splits ?
        self.uniform_mixing: bool = True
        self.lateral_context: int = 2  # TODO check default

    def _composite_patch(self, patches: Sequence[NDArray]) -> NDArray:
        if self.uniform_mixing:
            alphas = [1 / len(patches) for _ in range(len(patches))]
        else:
            alphas = []
            for i in range(len(patches)):
                alpha_pos = np.random.rand()
                alpha = self._start_alpha_arr[i] + alpha_pos * (
                    self._end_alpha_arr[i] - self._start_alpha_arr[i]
                )
                alphas.append(alpha)
        patch = sum(img * alpha for img, alpha in zip(patches, alphas))

        return patch

    def _get_patches(
        self, sample_idx: int, coords: Sequence[int], patch_size: Sequence[int]
    ) -> Sequence[NDArray]:
        patches_per_channel = []
        for channel_idx, coord_pair in enumerate(coords):
            patch = self._data[
                (
                    sample_idx,  # type: ignore
                    channel_idx,  # type: ignore
                    ...,  # type: ignore
                    *[slice(c, c + e) for c, e in zip(coord_pair, patch_size)],  # type: ignore
                )
            ]
            if not all(p == r for p, r in zip(patch.shape, patch_size)):
                continue  # TODO add padding
            patches_per_channel.append(patch)
        return self._composite_patch(patches_per_channel)

    def _get_lc_patches(
        self, sample_idx: int, coords: Sequence[int], patch_size: Sequence[int]
    ) -> Sequence[NDArray]:
        patches_per_hierarchy_level = []
        for context_level in range(1, self.lateral_context + 1):
            lc_coords = [
                (c - context_level * p // 2, e - context_level * p // 2)
                for (c, e), p in zip(coords, patch_size)
            ]
            lc_patch_size = [dim * context_level for dim in patch_size]
            patches_per_hierarchy_level.append(
                self._get_patches(
                    sample_idx=sample_idx, coords=lc_coords, patch_size=lc_patch_size
                )
            )
            print([p.shape for p in patches_per_hierarchy_level])
        return np.stack(patches_per_hierarchy_level)

    def extract_patch(
        self,
        sample_idx: int,
        coords: Sequence[int],
        patch_size: Sequence[int],
    ) -> NDArray:
        # TODO change?
        if not isinstance(coords, list):
            coords = [coords]
        # coords is a list of coordinates for each channel if we need to extract patches
        # from different spatial locations for each channel
        if any(len(c) != len(patch_size) for c in coords):
            raise ValueError("Length of coords and extent must match.")

        # TODO: test for 2D or 3D?
        if self.lateral_context > 1:
            patches = self._get_lc_patches(
                sample_idx=sample_idx, coords=coords, patch_size=patch_size
            )  # (lc, h, w)
        else:
            patches = self._get_patches(
                sample_idx=sample_idx, coords=coords, patch_size=patch_size
            )  # (1, h, w)

        return patches

    @classmethod
    def from_array(cls, data: NDArray, axes: str) -> Self:
        data = reshape_array(data, axes)
        return cls(source="array", data=data)

    @classmethod
    def from_tiff(cls, path: Path, axes: str) -> Self:
        data = read_tiff(path)
        data = reshape_array(data, axes)
        return cls(source=path, data=data)

    @classmethod
    def from_custom_file_type(
        cls, path: Path, axes: str, read_func: ReadFunc, **read_kwargs: Any
    ) -> Self:
        data = read_func(path, **read_kwargs)
        data = reshape_array(data, axes)
        return cls(source=path, data=data)
