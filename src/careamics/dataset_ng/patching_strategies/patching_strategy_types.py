from collections.abc import Sequence
from typing import Protocol, TypedDict


class PatchSpecs(TypedDict):
    data_idx: int
    sample_idx: int
    coords: Sequence[int]
    patch_size: Sequence[int]


class PatchingStrategy(Protocol):

    def get_patch_spec(self, index: int) -> PatchSpecs: ...
