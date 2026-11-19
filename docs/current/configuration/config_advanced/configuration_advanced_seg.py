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
