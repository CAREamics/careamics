#!/usr/bin/env python
# %%
# pre-requisite
import numpy as np

from careamics import CAREamist
from careamics.config import create_n2v_configuration
from careamics.utils import get_careamics_home

config = create_n2v_configuration(
    experiment_name="n2v_2D",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=1,
    num_epochs=1,
)

careamist = CAREamist(config)


train_array = np.random.rand(256, 256)
val_array = np.random.rand(256, 256)
my_array = np.random.rand(64, 64).astype(np.float32)

careamist.train(
    train_source=train_array,
    val_source=val_array,
)

export_path = get_careamics_home()

# %%
# --8<-- [start:export]
careamist.export_to_bmz(
    path_to_archive=export_path / "my_model.zip",  # (1)!
    friendly_model_name="CARE_mito",  # (2)!
    input_array=my_array,  # (3)!
    authors=[
        {
            "name": "Ignatius J. Reilly",
            "affiliation": "Levy Pants",
            "email": "ijr@levy.com",
        },
        {"name": "Myrna Minkoff", "orcid": "0000-0002-3291-8524"},  # (4)!
    ],
    general_description="This model was trained to denoise 2D images of mitochondria.",  # (5)!
    data_description="The data was acquired on a confocal microscope [...]",  # (6)!
)
# --8<-- [end:export]
# %%
# --8<-- [start:optional]
careamist.export_to_bmz(
    path_to_archive=export_path / "my_model.zip",
    friendly_model_name="CARE_mito",
    input_array=my_array,
    authors=[
        {
            "name": "Ignatius J. Reilly",
            "affiliation": "Levy Pants",
            "email": "ijr@levy.com",
        },
        {"name": "Myrna Minkoff", "orcid": "0000-0002-3291-8524"},
    ],
    general_description="This model was trained to denoise 2D images of mitochondria.",
    data_description="The data was acquired on a confocal microscope [...]",
    channel_names=["mito", "nucleus"],  # (1)!
)
# --8<-- [end:optional]
