"""Segmentation-specific configuration examples for the documentation."""

# --8<-- [start:seg_prediction]
from careamics.config import create_seg_config

config = create_seg_config(
    experiment_name="segmentation_probabilities",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    n_classes=2,
    prediction_output="probabilities",
)
# --8<-- [end:seg_prediction]

# --8<-- [start:seg_loss]
from careamics.config import create_advanced_seg_config

config = create_advanced_seg_config(
    experiment_name="segmentation_loss",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    n_classes=2,
    loss="dice_ce",
    loss_parameters={
        "class_weights": [1.0, 2.0, 3.0],
        "dice_weight": 0.5,
        "ce_weight": 0.5,
    },
)
# --8<-- [end:seg_loss]

# --8<-- [start:seg_filter]
from careamics.config import MaxPatchFilterConfig, create_advanced_seg_config

config = create_advanced_seg_config(
    experiment_name="segmentation_filtering",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    n_classes=2,
    patch_filter_config=MaxPatchFilterConfig(
        threshold=0.1,
        filtered_patch_prob=0.2,
    ),
)
# --8<-- [end:seg_filter]
