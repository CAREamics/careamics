"""Patch extractor that limits how many file stacks are loaded at once."""

from collections.abc import Sequence

from numpy.typing import NDArray

from ..image_stack import FileImageStack
from .patch_extractor import PatchExtractor


class LimitFilesPatchExtractor(PatchExtractor[FileImageStack]):
    """Patch extractor that limits how many file stacks are loaded at once.

    This patch extractor should be used when the data residing on disk is too large to
    be loaded in memory at once.

    Parameters
    ----------
    image_stacks : sequence of FileImageStack
        Image stacks to extract patches from.
    """

    def __init__(
        self,
        image_stacks: Sequence[FileImageStack],
    ) -> None:
        """Constructor.

        Parameters
        ----------
        image_stacks : sequence of FileImageStack
            Image stacks to extract patches from; only a subset are loaded at a time.
        """
        super().__init__(image_stacks)
        self.loaded_stacks: list[int] = []

    def extract_channel_patch(
        self,
        data_idx: int,
        sample_idx: int,
        channels: Sequence[int] | None,
        coords: Sequence[int],
        patch_size: Sequence[int],
    ) -> NDArray:
        """Extract patch, loading the file if not already loaded.

        Parameters
        ----------
        data_idx : int
            Image stack index.
        sample_idx : int
            Sample index.
        channels : sequence of int or None
            Channel indices; None for all.
        coords : sequence of int
            Patch start coordinates.
        patch_size : sequence of int
            Patch size per spatial dimension.

        Returns
        -------
        NDArray
            Patch data.
        """
        if data_idx not in self.loaded_stacks:
            # TODO: make maximum images loaded configurable?
            if len(self.loaded_stacks) >= 1:
                # get the idx that was added longest ago
                idx_to_close = self.loaded_stacks.pop(0)
                self.image_stacks[idx_to_close].close()

            self.image_stacks[data_idx].load()
            self.loaded_stacks.append(data_idx)

        return super().extract_channel_patch(
            data_idx, sample_idx, channels, coords, patch_size
        )
