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

    """

    # TODO: not sure how compatible using Path will be for a zarr array
    #   (for a zarr array need to specify file path and internal zarr path)
    @property
    def source(self) -> Union[str, Path, Literal["array"]]: ...

    @property
    def data_shape(self) -> Sequence[int]: ...

    @property
    def data_dtype(self) -> DTypeLike: ...

    def extract_patch(
        self, sample_idx: int, coords: Sequence[int], patch_size: Sequence[int]
    ) -> NDArray:
        """
        Extracts a patch for a given sample within the image stack.

        Parameters
        ----------
        sample_idx: int
            Sample index. The first dimension of the image data will be indexed at this
            value.
        coords: Sequence of int
            The coordinates that define the start of a patch.
        patch_size: Sequence of int
            The size of the patch in each spatial dimension.

        Returns
        -------
        numpy.ndarray
            A patch of the image data from a particlular sample. It will have the
            dimensions C(Z)YX.
        """
        ...


GenericImageStack = TypeVar("GenericImageStack", bound=ImageStack, covariant=True)
