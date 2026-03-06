#!/usr/bin/env python

# %%
# Noise2Void with channels
# --8<-- [start:simple]
from careamics.config import (
    create_care_configuration,  # CARE
    create_n2n_configuration,  # Noise2Noise
    create_n2v_configuration,  # Noise2Void
)

config = create_n2v_configuration(
    experiment_name="n2v_experiment",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=50,
)
# --8<-- [end:simple]


# %%
# Noise2Void with steps
# --8<-- [start:n2v_steps]
config = create_n2v_configuration(
    experiment_name="n2v_2D_steps",
    data_type="tiff",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=20,
    num_steps=50,  # (1)!
)
# --8<-- [end:n2v_steps]
# %%
# N2N with steps
# --8<-- [start:n2n_steps]
config = create_n2n_configuration(
    experiment_name="n2n_2D_steps",
    data_type="tiff",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=20,
    num_steps=50,  # (1)!
)
# --8<-- [end:n2n_steps]
# %%
# CARE with steps
# --8<-- [start:care_steps]
config = create_care_configuration(
    experiment_name="care_2D_steps",
    data_type="tiff",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=20,
    num_steps=50,  # (1)!
)
# --8<-- [end:care_steps]


# %%
# Noise2Void with channels
# --8<-- [start:n2v_channels]
config = create_n2v_configuration(
    experiment_name="n2v_2D_channels",
    data_type="tiff",
    axes="YXC",  # (1)!
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=20,
    n_channels=3,  # (2)!
)
# --8<-- [end:n2v_channels]
# %%
# Noise2Void channels together
# --8<-- [start:n2v_mix_channels]
config = create_n2v_configuration(
    experiment_name="n2v_2D_mix_channels",
    data_type="tiff",
    axes="YXC",  # (1)!
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=20,
    n_channels=3,
    independent_channels=False,  # (2)!
)
# --8<-- [end:n2v_mix_channels]
# %%
# N2V without augmentations
# --8<-- [start:n2v_no_aug]
config = create_n2v_configuration(
    experiment_name="n2v_2D_no_aug",
    data_type="tiff",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=20,
    augmentations=[],  # (1)!
)
# --8<-- [end:n2v_no_aug]
# N2V without augmentations
# --8<-- [start:n2v_aug]
from careamics.config.transformations import XYFlipConfig

config = create_n2v_configuration(
    experiment_name="n2v_2D_no_aug",
    data_type="tiff",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=20,
    augmentations=[XYFlipConfig(flip_y=False)],  # (1)!
)
# --8<-- [end:n2v_aug]
# --8<-- [start:care_aug]
from careamics.config.transformations import XYFlipConfig

config = create_care_configuration(
    experiment_name="care_2D_aug",
    data_type="tiff",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=20,
    augmentations=[XYFlipConfig(flip_y=False)],  # (1)!
)
# --8<-- [end:care_aug]
# --8<-- [start:n2n_aug]
from careamics.config.transformations import XYFlipConfig

config = create_n2n_configuration(
    experiment_name="n2n_2D_aug",
    data_type="tiff",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=20,
    augmentations=[XYFlipConfig(flip_y=False)],  # (1)!
)
# --8<-- [end:n2n_aug]
# %%
# N2V with WandB
# --8<-- [start:n2v_wandb]
config = create_n2v_configuration(
    experiment_name="n2v_2D_wandb",
    data_type="tiff",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=20,
    logger="wandb",  # (1)!
)
# --8<-- [end:n2v_wandb]
# %%
# N2V model specific parameters
# --8<-- [start:n2v_model_kwargs]
config = create_n2v_configuration(
    experiment_name="n2v_3D",
    data_type="tiff",
    axes="ZYX",
    patch_size=[16, 64, 64],
    batch_size=8,
    num_epochs=20,
    model_params={
        "depth": 3,  # (1)!
        "num_channels_init": 64,  # (2)!
        # (3)!
    },
)
# --8<-- [end:n2v_model_kwargs]
# %%
# N2V parameters
# --8<-- [start:n2v_parameters]
config = create_n2v_configuration(
    experiment_name="n2v_2D",
    data_type="tiff",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=20,
    roi_size=7,
    masked_pixel_percentage=0.5,
)
# --8<-- [end:n2v_parameters]
# %%
# N2V2
# --8<-- [start:n2v2]
config = create_n2v_configuration(
    experiment_name="n2v2_3D",
    data_type="tiff",
    axes="ZYX",
    patch_size=[16, 64, 64],
    batch_size=8,
    num_epochs=20,
    use_n2v2=True,  # (1)!
)
# --8<-- [end:n2v2]
# %%
# structN2V
# --8<-- [start:structn2v]
config = create_n2v_configuration(
    experiment_name="structn2v_3D",
    data_type="tiff",
    axes="ZYX",
    patch_size=[16, 64, 64],
    batch_size=8,
    num_epochs=20,
    struct_n2v_axis="horizontal",
    struct_n2v_span=5,
)
# --8<-- [end:structn2v]

# %%
# N2N multiple channels
# --8<-- [start:n2n_channels]
config = create_n2n_configuration(
    experiment_name="n2n_2D_channels",
    data_type="tiff",
    axes="YXC",  # (1)!
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=20,
    n_channels_in=3,  # (2)!
    n_channels_out=2,  # (3)!
)

# --8<-- [end:n2n_channels]
# %%
# N2N channels together
# --8<-- [start:n2n_mix_channels]
config = create_n2n_configuration(
    experiment_name="n2n_2D_mix_channels",
    data_type="tiff",
    axes="YXC",  # (1)!
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=20,
    n_channels_in=3,
    independent_channels=False,  # (2)!
)
# --8<-- [end:n2n_mix_channels]

# %%
# N2N without augmentations
# --8<-- [start:n2n_no_aug]
config = create_n2n_configuration(
    experiment_name="n2n_2D_no_aug",
    data_type="tiff",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=20,
    augmentations=[],  # (1)!
)
# --8<-- [end:n2n_no_aug]

# %%
# N2N with WandB
# --8<-- [start:n2n_wandb]
config = create_n2n_configuration(
    experiment_name="n2n_2D_wandb",
    data_type="tiff",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=20,
    logger="wandb",  # (1)!
)
# --8<-- [end:n2n_wandb]
# %%
# N2N with model specific parameters
# --8<-- [start:n2n_model_kwargs]
config = create_n2n_configuration(
    experiment_name="n2n_3D",
    data_type="tiff",
    axes="ZYX",
    patch_size=[16, 64, 64],
    batch_size=8,
    num_epochs=20,
    model_params={
        "depth": 3,  # (1)!
        "num_channels_init": 64,  # (2)!
        # (3)!
    },
)
# --8<-- [end:n2n_model_kwargs]

# %%
# N2N with different loss
# --8<-- [start:n2n_loss]
config = create_n2n_configuration(
    experiment_name="n2n_3D",
    data_type="tiff",
    axes="ZYX",
    patch_size=[16, 64, 64],
    batch_size=8,
    num_epochs=20,
    loss="mae",  # (1)!
)
# --8<-- [end:n2n_loss]
# %%
# CARE with multiple channels
# --8<-- [start:care_channels]
config = create_care_configuration(
    experiment_name="care_2D_channels",
    data_type="tiff",
    axes="YXC",  # (1)!
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=20,
    n_channels_in=3,  # (2)!
    n_channels_out=2,  # (3)!
)
# --8<-- [end:care_channels]

# %%
# CARE channels together
# --8<-- [start:care_mix_channels]
config = create_care_configuration(
    experiment_name="care_2D_mix_channels",
    data_type="tiff",
    axes="YXC",  # (1)!
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=20,
    n_channels_in=3,
    n_channels_out=2,
    independent_channels=False,  # (2)!
)
# --8<-- [end:care_mix_channels]
# %%
# CARE without augmentations
# --8<-- [start:care_no_aug]
config = create_care_configuration(
    experiment_name="care_2D_no_aug",
    data_type="tiff",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=20,
    augmentations=[],  # (1)!
)
# --8<-- [end:care_no_aug]

# %%
# CARE with WandB
# --8<-- [start:care_wandb]
config = create_care_configuration(
    experiment_name="care_2D_wandb",
    data_type="tiff",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=20,
    logger="wandb",  # (1)!
)
# --8<-- [end:care_wandb]
# %%
# CARE with model specific parameters
# --8<-- [start:care_model_kwargs]
config = create_care_configuration(
    experiment_name="care_3D",
    data_type="tiff",
    axes="ZYX",
    patch_size=[16, 64, 64],
    batch_size=8,
    num_epochs=20,
    model_params={
        "depth": 3,  # (1)!
        "num_channels_init": 64,  # (2)!
        # (3)!
    },
)
# --8<-- [end:care_model_kwargs]
# %%
# CARE with different loss
# --8<-- [start:care_loss]
config = create_care_configuration(
    experiment_name="care_3D",
    data_type="tiff",
    axes="ZYX",
    patch_size=[16, 64, 64],
    batch_size=8,
    num_epochs=20,
    loss="mae",  # (1)!
)
# --8<-- [end:care_loss]


# %%
# N2N with dataloader parameters
# --8<-- [start:n2v_dataloader_kwargs]
config = create_n2v_configuration(
    experiment_name="n2v_3D",
    data_type="tiff",
    axes="ZYX",
    patch_size=[16, 64, 64],
    batch_size=8,
    num_epochs=20,
    train_dataloader_params={
        "num_workers": 4,  # (1)!
    },
    val_dataloader_params={
        "num_workers": 2,  # (2)!
    },
)
# --8<-- [end:n2v_dataloader_kwargs]
# N2N with dataloader parameters
# --8<-- [start:care_dataloader_kwargs]
config = create_care_configuration(
    experiment_name="n2n_3D",
    data_type="tiff",
    axes="ZYX",
    patch_size=[16, 64, 64],
    batch_size=8,
    num_epochs=20,
    train_dataloader_params={
        "num_workers": 4,  # (1)!
    },
    val_dataloader_params={
        "num_workers": 2,  # (2)!
    },
)
# --8<-- [end:care_dataloader_kwargs]
# N2N with dataloader parameters
# --8<-- [start:n2n_dataloader_kwargs]
config = create_n2n_configuration(
    experiment_name="n2n_3D",
    data_type="tiff",
    axes="ZYX",
    patch_size=[16, 64, 64],
    batch_size=8,
    num_epochs=20,
    train_dataloader_params={
        "num_workers": 4,  # (1)!
    },
    val_dataloader_params={
        "num_workers": 2,  # (2)!
    },
)
# --8<-- [end:n2n_dataloader_kwargs]
