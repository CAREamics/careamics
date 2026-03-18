#!/usr/bin/env python
# %%
# --8<-- [start:basic_usage]
import numpy as np

from careamics import CAREamist
from careamics.config import create_n2v_configuration

# create a configuration
config = create_n2v_configuration(
    experiment_name="n2v_2D",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=1,
    num_epochs=1,  # (1)!
)

# instantiate a careamist
careamist = CAREamist(config)

# train the model
train_data = np.random.randint(0, 255, (256, 256))  # (2)!
careamist.train(train_source=train_data)

# once trained, predict
pred_data = np.random.randint(0, 255, (128, 128)).astype(np.float32)
predction = careamist.predict(source=pred_data)

# export to BMZ format
careamist.export_to_bmz(
    path_to_archive="n2v_model.zip",
    friendly_model_name="N2V 2D",
    input_array=pred_data,
    authors=[{"name": "CAREamics authors"}],
    general_description="This model was trained to denoise 2D images.",
    data_description="The data was acquired on a confocal microscope [...]",
)
# --8<-- [end:basic_usage]
