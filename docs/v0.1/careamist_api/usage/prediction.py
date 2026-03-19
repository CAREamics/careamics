#!/usr/bin/env python
# %%
# pre-requisite
from pathlib import Path

import numpy as np

from careamics import CAREamist
from careamics.config import create_n2v_configuration
from careamics.utils import get_careamics_home

train_array = np.random.rand(256, 256)
val_array = np.random.rand(256, 256)

config = create_n2v_configuration(
    experiment_name="n2v_2D",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=1,
    num_epochs=1,
)

workdir = get_careamics_home()

careamist = CAREamist(
    config,
    work_dir=workdir,
)

careamist.train(
    train_source=train_array,
    val_source=val_array,
)


# %%
# --8<-- [start:array]
import numpy as np

array = np.random.rand(256, 256)

careamist.predict(
    source=array,
)
# --8<-- [end:array]

# %%
import tifffile

path_to_data = get_careamics_home() / "test" / "data.tiff"
path_to_data.parent.mkdir(exist_ok=True, parents=True)
tifffile.imwrite(path_to_data, array)
assert path_to_data.exists()

config.data_config.data_type = "tiff"

# --8<-- [start:path]
careamist.predict(
    source=path_to_data,
)
# --8<-- [end:path]

# %%
config.data_config.data_type = "array"
# --8<-- [start:tiling]
careamist.predict(
    source=array,
    tile_size=[64, 64],  # (1)!
    tile_overlap=[32, 32],  # (2)!
)
# --8<-- [end:tiling]

# %%
# --8<-- [start:tta]
careamist.predict(
    source=array,
    tta=False,  # (1)!
)
# --8<-- [end:tta]

# %%
# --8<-- [start:batches]
careamist.predict(
    source=array,
    batch_size=2,  # (1)!
)
# --8<-- [end:batches]

# %%
# --8<-- [start:diff_type]
careamist.predict(
    source=path_to_data,
    data_type="tiff",  # (1)!
)
# --8<-- [end:diff_type]

# %%
other_array = np.random.rand(3, 256, 256)

# --8<-- [start:diff_axes]
careamist.predict(
    source=other_array,
    axes="SYX",  # (1)!
)
# --8<-- [end:diff_axes]

# %%
from typing import Any


# --8<-- [start:custom_type]
def read_npy(
    path: Path,
    *args: Any,
    **kwargs: Any,
) -> np.ndarray:
    return np.load(path)


# example data
predict_array = np.random.rand(128, 128)
np.save("train_array.npy", train_array)

# Train
careamist.predict(
    source=predict_array,
    read_source_func=read_npy,
    extension_filter="*.npy",
)
# --8<-- [end:custom_type]
