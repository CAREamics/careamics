from collections.abc import Sequence
from pathlib import Path
from typing import Optional, TypedDict

import numpy as np
from numpy.typing import NDArray

from .in_memory_array_reader import InMemoryArrayReader
from .protocol import ProtoArrayReader


class PatchSpecs(TypedDict):
    data_idx: int
    sample_idx: int
    coords: tuple[int, ...]
    extent: tuple[int, ...]


# TODO: bad name?
class PatchExtractor:

    def __init__(self, data_readers: Sequence[ProtoArrayReader]):
        self.data_readers: list[ProtoArrayReader] = list(data_readers)

    @classmethod
    def from_arrays(cls, arrays: Sequence[NDArray], axes: str):
        data_readers = [
            InMemoryArrayReader.from_array(data=array, axes=axes) for array in arrays
        ]
        return cls(data_readers=data_readers)

    @classmethod
    def from_tiff_files(cls, file_paths: Sequence[Path], axes: str):
        data_readers = [
            InMemoryArrayReader.from_tiff(path=path, axes=axes) for path in file_paths
        ]
        return cls(data_readers=data_readers)

    def extract_patch(
        self,
        data_idx: int,
        sample_idx: int,
        coords: tuple[int, ...],
        extent: tuple[int, ...],
    ) -> NDArray:
        return self.data_readers[data_idx].extract_patch(
            sample_idx=sample_idx, coords=coords, extent=extent
        )

    def extract_patches(self, patch_specs: Sequence[PatchSpecs]) -> list[NDArray]:
        return [self.extract_patch(**patch_spec) for patch_spec in patch_specs]

    # TODO: maybe don't want this as an instance method
    #   - instead pass datashapes to an external function
    #   - (Thinking about when we also want to create patches from a target)
    def generate_random_patch_specs(
        self, patch_size: tuple[int, ...], seed: Optional[int] = None
    ) -> Sequence[PatchSpecs]:
        rng = np.random.default_rng(seed=seed)
        patch_specs: list[PatchSpecs] = []
        for data_idx, data_reader in enumerate(self.data_readers):

            # shape on which data is patched
            data_spatial_shape = data_reader.data_shape[-len(patch_size) :]

            n_patches = int(np.ceil(np.prod(data_spatial_shape) / np.prod(patch_size)))
            data_patch_specs = [
                PatchSpecs(
                    data_idx=data_idx,
                    sample_idx=sample_idx,
                    coords=tuple(
                        rng.integers(
                            np.zeros(len(patch_size), dtype=int),
                            np.array(data_spatial_shape) - np.array(patch_size),
                            endpoint=True,
                        )
                    ),
                    extent=patch_size,
                )
                for sample_idx in range(data_reader.data_shape[0])
                for _ in range(n_patches)
            ]
            patch_specs.extend(data_patch_specs)
        return patch_specs
