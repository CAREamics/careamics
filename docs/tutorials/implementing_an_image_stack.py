# %%
# --8<-- [start:hdf5-image-stack]
from collections.abc import Sequence
import h5py

from numpy.typing import NDArray, DTypeLike

from careamics.utils.reshape_array import reshape_patch, get_patch_slices, AxesTransform
from careamics.dataset.image_stack.image_utils import pad_patch


class HDF5ImageStack:
    def __init__(self, image_data: h5py.Dataset, axes: str):
        self._image_data = image_data
        self.original_axes = axes
        self.original_data_shape = image_data.shape
        self.data_shape = AxesTransform(
            axes, self.original_data_shape
        ).transformed_shape

    @property
    def data_dtype(self) -> DTypeLike:
        return self._image_data.dtype

    @property
    def source(self) -> str:  # (1)!
        return "#".join([self._image_data.file.filename, str(self._image_data.name)])

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
        patch_slice = get_patch_slices(
            self.original_axes,
            self.original_data_shape,
            sample_idx,
            channels,
            coords,
            patch_size,
        )
        patch_data = self._image_data[patch_slice]  # (2)!
        patch_data = reshape_patch(patch_data, self.original_axes)
        patch = pad_patch(coords, patch_size, self.data_shape, patch_data)
        return patch


# --8<-- [end:hdf5-image-stack]

# %%
# --8<-- [start:transform-data-shape]
from careamics.utils.reshape_array import AxesTransform

original_axes = "YXC"
original_data_shape = (512, 620, 2)
data_shape = AxesTransform(original_axes, original_data_shape).transformed_shape
print(data_shape)
# --8<-- [end:transform-data-shape]
# %%
