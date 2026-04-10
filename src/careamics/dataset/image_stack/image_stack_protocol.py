"""Image stack protocol and type variables."""

from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Protocol, TypeVar, Union

from numpy.typing import DTypeLike, NDArray


class ImageStack(Protocol):
    """
    An interface for extracting patches from an image stack.

    Attributes
    ----------
    source: Path or "array"
        Origin of the image data.
    data_shape: Sequence[int]
        The shape of the data, it is expected to be in the order (SC(Z)YX).
    data_dtype: DTypeLike
        The data type of the image data.
    """

    @property
    def source(self) -> Union[str, Path, Literal["array"]]:
        """Source of the image data."""
        ...

    @property
    def data_shape(self) -> Sequence[int]:
        """Shape of the image data (SC(Z)YX)."""
        ...

    @property
    def data_dtype(self) -> DTypeLike:
        """Data type of the image data."""
        ...

    @property
    def original_data_shape(self) -> Sequence[int]:
        """Original shape of the data."""
        ...

    @property
    def original_axes(self) -> str:
        """Original axes of the data."""
        ...

    def extract_patch(
        self,
        sample_idx: int,
        channels: Sequence[int] | None,
        coords: Sequence[int],
        patch_size: Sequence[int],
    ) -> NDArray:
        """Extract a patch for a given sample and channels within the image stack.

        Parameters
        ----------
        sample_idx : int
            Sample index.
        channels : sequence of int or None
            Channel indices to extract. If `None`, all channels will be extracted.
        coords : sequence of int
            Spatial coordinates of the top-left corner of the patch.
        patch_size : sequence of int
            Size of the patch in each spatial dimension.

        Returns
        -------
        numpy.ndarray
            A patch of the image data from a particular sample with dimensions C(Z)YX.
        """
        ...


GenericImageStack = TypeVar("GenericImageStack", bound=ImageStack, covariant=True)
