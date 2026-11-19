"""Advanced segmentation configuration examples for the documentation."""

# --8<-- [start:adv_config_seg]
from careamics.config import create_advanced_seg_config

config = create_advanced_seg_config(
    experiment_name="segmentation_training",
    data_type="array",
    axes="CYX",
    patch_size=[64, 64],
    batch_size=8,
    n_classes=2,
    channels=[0, 2],
    normalization="mean_std",
    loss="dice_ce",
    num_epochs=30,
)
# --8<-- [end:adv_config_seg]

# %%
# --8<-- [start:adv_config_seg_in_memory]
from careamics.config import create_advanced_seg_config

config = create_advanced_seg_config(
    experiment_name="segmentation_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    n_classes=2,
    in_memory=True,  # (1)!
)
# --8<-- [end:adv_config_seg_in_memory]

# %%
# --8<-- [start:adv_config_seg_subchannels]
from careamics.config import create_advanced_seg_config

config = create_advanced_seg_config(
    experiment_name="segmentation_training",
    data_type="array",
    axes="CYX",
    patch_size=[64, 64],
    batch_size=8,
    n_classes=2,
    channels=[0, 2],  # (1)!
)
# --8<-- [end:adv_config_seg_subchannels]

# %%
# --8<-- [start:adv_config_seg_norm]
from careamics.config import create_advanced_seg_config

config = create_advanced_seg_config(
    experiment_name="segmentation_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    n_classes=2,
    normalization="quantile",  # (1)!
    normalization_params={"lower_quantiles": [0.01], "upper_quantiles": [0.99]},  # (2)!
)
# --8<-- [end:adv_config_seg_norm]

# %%
# --8<-- [start:adv_config_seg_loss]
from careamics.config import create_advanced_seg_config

config = create_advanced_seg_config(
    experiment_name="segmentation_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    n_classes=2,
    loss="dice_ce",  # (1)!
    loss_parameters={
        "class_weights": [1.0, 2.0, 3.0],
        "dice_weight": 0.5,
        "ce_weight": 0.5,
    },  # (2)!
)
# --8<-- [end:adv_config_seg_loss]

# %%
# --8<-- [start:adv_config_seg_filter]
from careamics.config import MaxPatchFilterConfig, create_advanced_seg_config

config = create_advanced_seg_config(
    experiment_name="segmentation_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    n_classes=2,
    patch_filter_config=MaxPatchFilterConfig(
        threshold=0.1, filtered_patch_prob=0.2
    ),  # (1)!
)
# --8<-- [end:adv_config_seg_filter]

# %%
# --8<-- [start:adv_config_seg_model]
from careamics.config import create_advanced_seg_config

config = create_advanced_seg_config(
    experiment_name="segmentation_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    n_classes=2,
    model_params={"depth": 3, "num_channels_init": 64, "use_batch_norm": True},  # (1)!
)
# --8<-- [end:adv_config_seg_model]

# %%
# --8<-- [start:adv_config_seg_num_workers]
from careamics.config import create_advanced_seg_config

config = create_advanced_seg_config(
    experiment_name="segmentation_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    n_classes=2,
    num_workers=4,  # (1)!
)
# --8<-- [end:adv_config_seg_num_workers]

# %%
# --8<-- [start:adv_config_seg_seed]
from careamics.config import create_advanced_seg_config

config = create_advanced_seg_config(
    experiment_name="segmentation_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    n_classes=2,
    seed=42,
)
# --8<-- [end:adv_config_seg_seed]

# %%
# --8<-- [start:adv_config_seg_logger]
from careamics.config import create_advanced_seg_config

config = create_advanced_seg_config(
    experiment_name="segmentation_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    n_classes=2,
    logger="wandb",  # (1)!
)
# --8<-- [end:adv_config_seg_logger]
