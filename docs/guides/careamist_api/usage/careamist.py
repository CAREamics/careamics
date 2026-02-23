#!/usr/bin/env python
# %%
# --8<-- [start:config]
from careamics import CAREamist
from careamics.config import create_n2v_configuration

config = create_n2v_configuration(
    experiment_name="n2v_2D",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=20,
)  # (1)!

careamist = CAREamist(config)
# --8<-- [end:config]

# %%
# --8<-- [start:config_path]
from careamics import CAREamist
from careamics.config import create_n2v_configuration, save_configuration

config = create_n2v_configuration(
    experiment_name="n2v_2D",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=20,
)

save_configuration(config, "configuration_example.yml")

careamist = CAREamist("configuration_example.yml")
# --8<-- [end:config_path]

# %%
# necessary to export to the BMZ (pretending it trained)
careamist.cfg.data_config.set_means_and_stds([0.0], [1.0])

# %%
import numpy as np

careamist.export_to_bmz(
    path_to_archive="model.zip",
    friendly_model_name="MyExampleModel",
    input_array=np.random.randint(0, 255, (128, 128)).astype(np.float32),
    authors=[{"name": "CAREamics"}],
    general_description="This model was trained to denoise 2D images.",
    data_description="The data was acquired on a confocal microscope [...]",
)

# %%
# --8<-- [start:load_model]
from careamics import CAREamist

path_to_model = "model.zip"  # (1)!

careamist = CAREamist(path_to_model)
# --8<-- [end:load_model]


# %%
# --8<-- [start:work_dir]
careamist = CAREamist(config, work_dir="work_dir")
# --8<-- [end:work_dir]

# %%
# Callbacks
# --8<-- [start:callbacks]
from pytorch_lightning.callbacks import Callback


# define a custom callback
class MyCallback(Callback):  # (1)!
    def __init__(self):
        super().__init__()

        self.has_started = False
        self.has_ended = False

    def on_train_start(self, trainer, pl_module):
        self.has_started = True  # (2)!

    def on_train_end(self, trainer, pl_module):
        self.has_ended = True


my_callback = MyCallback()  # (3)!

careamist = CAREamist(config, callbacks=[my_callback])  # (4)!
# --8<-- [end:callbacks]
