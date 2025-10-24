from collections.abc import Sequence

from numpy.typing import NDArray

from .image_stack import FileImageStack
from .patch_extractor import PatchExtractor


class LimitFilesPatchExtractor(PatchExtractor):

    def __init__(self, image_stacks: Sequence[FileImageStack]):
        self.image_stacks: list[FileImageStack]
        super().__init__(image_stacks)
        self.loaded_stacks: list[int] = []

    def extract_patch(
        self,
        data_idx: int,
        sample_idx: int,
        coords: Sequence[int],
        patch_size: Sequence[int],
    ) -> NDArray:
        if data_idx not in self.loaded_stacks:
            self.image_stacks[data_idx].load()
            self.loaded_stacks.append(data_idx)
        # TODO: make maximum images loaded configurable?
        if len(self.loaded_stacks) > 2:
            # get the idx that was added longest ago
            idx_to_close = self.loaded_stacks.pop(0)
            self.image_stacks[idx_to_close].close()

        return super().extract_patch(data_idx, sample_idx, coords, patch_size)
