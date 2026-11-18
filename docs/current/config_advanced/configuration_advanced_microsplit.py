"""Advanced MicroSplit configuration examples for the documentation."""

# --8<-- [start:adv_config_microsplit]
from careamics.config import create_advanced_microsplit_config

config = create_advanced_microsplit_config(
    experiment_name="microsplit_training",
    data_type="array",
    axes="CYX",
    patch_size=[64, 64],
    batch_size=8,
    output_channels=1,
    channels=[0, 2],
    multiscale_count=3,
    padding_mode="reflect",
    normalization="mean_std",
    optimizer="Adamax",
    optimizer_params={"lr": 1e-3},
    num_epochs=100,
)
# --8<-- [end:adv_config_microsplit]
