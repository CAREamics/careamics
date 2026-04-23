# %%
# --8<-- [start:data-config]
from collections.abc import Sequence
from pathlib import Path
import h5py

import numpy as np
from numpy.typing import DTypeLike, NDArray

from careamics import CAREamist, ImageStackLoading
from careamics.config import create_n2v_config
from careamics.utils.reshape_array import reshape_patch, get_patch_slices, AxesTransform
from careamics.dataset.image_stack.image_utils import pad_patch

DATA_PATH = Path("data")

# --- create toy data
n_files = 5
image_shape = (512, 512)
hdf5_path = DATA_PATH / "dataset.h5"
with h5py.File(hdf5_path, "w") as hdf5_file:
    for i in range(5):
        image = np.random.rand(*image_shape)
        data_path = f"image_{i}"
        hdf5_file.create_dataset(name=data_path, data=image)
# HDF5 file with 5 image datasets at the root
# dataset.h5/
# ├── image_0
# ├── image_1
# ├── image_2
# ├── image_3
# └── image_4

# --- configuration
config = create_n2v_config(
    "loading-custom",
    data_type="custom",  # (1)!
    axes="YX",  # (2)!
    patch_size=(64, 64),
    batch_size=16,
    num_epochs=10,
)
# --8<-- [end:data-config]


# --8<-- [start:image-stack-loader]
# --- custom ImageStack, that adheres to the ImageStack protocol
# Adapted from the careamics native ZarrImageStack
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

        patch_data = self._image_data[patch_slice]  # type: ignore
        patch_data = reshape_patch(patch_data, self.original_axes)
        patch = pad_patch(coords, patch_size, self.data_shape, patch_data)
        return patch


# helper function
def _walk_hdf5(group: h5py.Group):
    """Iterate through every dataset contained in a HDF5 group"""
    keys = group.keys()
    for key in keys:
        node = group.get(key)
        if isinstance(node, h5py.Dataset):
            yield node
        elif isinstance(node, h5py.Group):
            yield from _walk_hdf5(node)
    return


# --- Define the loading function, adhering to the ImageStackLoader protocol
# NOTE: this is just one way to define a HDF5 loader, it could be adapted to:
# - Load from a list of HDF5 files, or
# - Load from a subset HDF5 groups within the file.
def load_hdf5s(source: h5py.File, axes: str) -> list[HDF5ImageStack]:  # (2)!
    """
    Load all the images in a HDF5 file.

    source : Path
        The HDF5 file.
    axes : str
        Axes order of the data (e.g. "SYX", "YXC").
    """
    image_stacks: list[HDF5ImageStack] = []
    for image_data in _walk_hdf5(source):
        image_stacks.append(HDF5ImageStack(image_data, axes))
    return image_stacks


# --8<-- [end:image-stack-loader]

# %%
# --8<-- [start:train-pred]
careamist = CAREamist(config)

# --- train
hdf5_file = h5py.File(hdf5_path)  # keep a reference to the open HDF5 file
careamist.train(
    train_data=hdf5_file,  # (1)!
    loading=ImageStackLoading(load_hdf5s),  # (2)!
)

# --- predict
prediction, sources = careamist.predict(
    pred_data=hdf5_file,
    loading=ImageStackLoading(load_hdf5s),
)
hdf5_file.close()  # close the file once done, see h5py docs
sources  # inspect the sources of the predictions # (3)!
# --8<-- [end:train-pred]

# %%
