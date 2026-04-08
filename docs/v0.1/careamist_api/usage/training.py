#!/usr/bin/env python
# %%
# pre-requisite
from careamics import CAREamist
from careamics.config import create_n2v_configuration

config = create_n2v_configuration(
    experiment_name="n2v_2D",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=1,
    num_epochs=1,
)

careamist = CAREamist(config)

# %%
# --8<-- [start:array]
import numpy as np

train_array = np.random.rand(256, 256)
val_array = np.random.rand(256, 256)

careamist.train(
    train_source=train_array,  # (1)!
    val_source=val_array,  # (2)!
)
# --8<-- [end:array]

# %%
import tifffile

from careamics.utils import get_careamics_home

path_to_train_data = get_careamics_home() / "test" / "train_data.tiff"
path_to_train_data.parent.mkdir(exist_ok=True, parents=True)
tifffile.imwrite(path_to_train_data, train_array)
assert path_to_train_data.exists()

path_to_val_data = get_careamics_home() / "test" / "val_data.tiff"
path_to_val_data.parent.mkdir(exist_ok=True, parents=True)
tifffile.imwrite(path_to_val_data, train_array)
assert path_to_val_data.exists()

config.data_config.data_type = "tiff"

# --8<-- [start:path]
careamist.train(
    train_source=path_to_train_data,  # (1)!
    val_source=path_to_val_data,
)
# --8<-- [end:path]

# %%
config.data_config.data_type = "array"

# --8<-- [start:split]
careamist.train(
    train_source=train_array,
    val_percentage=0.1,  # (1)!
    val_minimum_split=5,  # (2)!
)
# --8<-- [end:split]

# %%
# --8<-- [start:datamodule]
from careamics.lightning import TrainDataModule

data_module = TrainDataModule(  # (1)!
    data_config=config.data_config, train_data=train_array
)

careamist.train(datamodule=data_module)
# --8<-- [end:datamodule]

# %%
from careamics.config import create_care_configuration

config_care = create_care_configuration(
    experiment_name="care_2D",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=1,
    num_epochs=1,
)

careamist_supervised = CAREamist(config_care)

target_array = np.random.rand(256, 256)
val_target_array = np.random.rand(256, 256)
# --8<-- [start:supervised]
careamist_supervised.train(
    train_source=train_array,
    train_target=target_array,
    val_source=val_array,
    val_target=val_target_array,
)
# --8<-- [end:supervised]


# %%

# --8<-- [start:losses]
loss_dict = careamist.get_losses()

from matplotlib import pyplot as plt

plt.plot(
    loss_dict["train_epoch"],
    loss_dict["train_loss"],
    loss_dict["val_epoch"],
    loss_dict["val_loss"],
)
plt.legend(["Train loss", "Val loss"])
plt.title("Losses")
# --8<-- [end:losses]
plt.close()
