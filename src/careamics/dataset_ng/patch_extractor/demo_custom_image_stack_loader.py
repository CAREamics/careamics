# %%
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
import zarr
from numpy.typing import NDArray
from zarr.storage import FSStore

from careamics.config import DataConfig
from careamics.config.support import SupportedData
from careamics.dataset_ng.patch_extractor.image_stack import ZarrImageStack
from careamics.dataset_ng.patch_extractor.image_stack_loader import ImageStackLoader
from careamics.dataset_ng.patch_extractor.patch_extractor_factory import (
    create_custom_image_stack_extractor,
)


# %%
def create_zarr_array(file_path: Path, data_path: str, data: NDArray):
    store = FSStore(url=file_path.resolve())
    # create array
    array = zarr.create(
        store=store,
        shape=data.shape,
        chunks=data.shape,  # only 1 chunk
        dtype=np.uint16,
        path=data_path,
    )
    # write data
    array[...] = data
    store.close()


def create_zarr(file_path: Path, data_paths: Sequence[str], data: Sequence[NDArray]):
    for data_path, array in zip(data_paths, data, strict=False):
        create_zarr_array(file_path=file_path, data_path=data_path, data=array)


# %% [markdown]
# ### Create example ZARR file

# %%
dir_path = Path("/home/melisande.croft/Documents/Data")
file_name = "test_ngff_image.zarr"
file_path = dir_path / file_name

data_paths = [
    "image_1",
    "group_1/image_1.1",
    "group_1/image_1.2",
]
data_shapes = [(1, 3, 64, 64), (1, 3, 32, 48), (1, 3, 32, 32)]
data = [np.random.randint(1, 255, size=shape, dtype=np.uint8) for shape in data_shapes]
if not file_path.is_file() and not file_path.is_dir():
    create_zarr(file_path, data_paths, data)

# %% [markdown]
# ### Make sure file exists

# %%
store = FSStore(url=file_path.resolve(), mode="r")

# %%
list(store.keys())

# %% [markdown]
# ### Define custom loading function


# %%
class ZarrSource(TypedDict):
    store: FSStore
    data_paths: Sequence[str]


def custom_image_stack_loader(source: ZarrSource, axes: str, *args, **kwargs):
    image_stacks = [
        ZarrImageStack(store=source["store"], data_path=data_path, axes=axes)
        for data_path in source["data_paths"]
    ]
    return image_stacks


# %% [markdown]
# ### Test custom loading func

# %%
# dummy data config
data_config = DataConfig(data_type="custom", patch_size=[64, 64], axes="SCYX")

# %%
image_stack_loader: ImageStackLoader = custom_image_stack_loader

# %%
# So pylance knows that datatype is custom to match function overloads
assert SupportedData(data_config.data_type) is SupportedData.CUSTOM

patch_extractor = create_custom_image_stack_extractor(
    source={"store": store, "data_paths": data_paths},
    axes=data_config.axes,
    data_type=SupportedData(data_config.data_type),
    image_stack_loader=custom_image_stack_loader,
)

# %%
# extract patch and display
patch = patch_extractor.extract_patch(2, 0, (8, 16), (16, 16))
plt.imshow(np.moveaxis(patch, 0, -1))

# %%
