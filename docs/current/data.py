#!/usr/bin/env python

# %%
# --8<-- [start:czi]
from careamics.config.factories import create_n2v_config

# create a configuration
config = create_n2v_config(
    experiment_name="n2v_czi",
    data_type="czi",  # (1)!
    axes="SCZYX",  # (2)!
    patch_size=[16, 64, 64],  # (3)!
    batch_size=8,
    num_epochs=30,
    n_channels=1,  # (4)!
)
# --8<-- [end:czi]

######################################## Zarr

# TODO use tempdir?
# %%
# --8<-- [start:zarr]
from pathlib import Path
import numpy as np
import zarr

# create a toy example
# - train_data.zarr"
#     - root_array_1: (128, 128)
#     - root_array_2: (128, 128)
#     - others
#         - array_2: (128, 128)
#         - array_3: (128, 128)
zarr_path = Path("train_data.zarr")
zarr_file = zarr.open(zarr_path, mode="w")
array_root_1 = zarr_file.create_array("array_1", data=np.random.rand(128, 128))
array_root_2 = zarr_file.create_array("array_2", data=np.random.rand(128, 128))

group = zarr_file.create_group("others")
array_2 = group.create_array("array_2", data=np.random.rand(128, 128))
array_3 = group.create_array("array_3", data=np.random.rand(128, 128))

# different ways to specify training data
train_from_zarr = zarr_path  # (1)!
train_from_group = str(group.store_path)  # (2)!
train_from_array = str(array_root_1.store_path)  # (3)!
train_from_list = [  # (4)!
    str(array_root_1.store_path),
    str(array_root_2.store_path),
    str(array_2.store_path),
    str(array_3.store_path),
]
# --8<-- [end:zarr]

# test that the training data can be read
from careamics.dataset_ng.image_stack_loader import load_zarrs

assert (
    len(load_zarrs([train_from_zarr], axes="YX")) == 2
), "Loading from zarr file failed."
assert (
    len(load_zarrs([train_from_group], axes="YX")) == 2
), "Loading from zarr group failed."
assert (
    len(load_zarrs([train_from_array], axes="YX")) == 1
), "Loading from zarr array failed."
assert (
    len(load_zarrs(train_from_list, axes="YX")) == 4
), "Loading from zarr list failed."
