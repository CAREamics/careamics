"""CryoCARE-specific configuration examples from the dev/cryocare branch."""

# --8<-- [start:cryocare_tilt_split]
from careamics.config import create_advanced_cryocare_config

config = create_advanced_cryocare_config(
    experiment_name="cryocare_tilt_split",
    data_type="mrc",
    axes="ZYX",
    patch_size=[16, 64, 64],
    batch_size=8,
    tilt_axis="Y",
    validation_fraction=0.2,
    n_val_patches=50,
)
# --8<-- [end:cryocare_tilt_split]

# --8<-- [start:cryocare_normalization]
config = create_advanced_cryocare_config(
    experiment_name="cryocare_normalization",
    data_type="mrc",
    axes="ZYX",
    patch_size=[16, 64, 64],
    batch_size=8,
    normalization="quantile",
    normalization_params={"lower_quantiles": [0.01], "upper_quantiles": [0.99]},
)
# --8<-- [end:cryocare_normalization]

# --8<-- [start:cryocare_validation]
from careamics.config import create_cryocare_config

config = create_cryocare_config(
    experiment_name="cryocare_validation",
    data_type="mrc",
    axes="ZYX",
    patch_size=[16, 64, 64],
    batch_size=8,
    tilt_axis="Y",
    validation_fraction=0.2,
)
# --8<-- [end:cryocare_validation]
