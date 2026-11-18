"""MicroSplit-specific configuration examples for the documentation."""

# --8<-- [start:microsplit_synthetic_inputs]
from careamics.config import create_advanced_microsplit_config

config = create_advanced_microsplit_config(
    experiment_name="microsplit_synthetic_inputs",
    data_type="array",
    axes="CYX",
    patch_size=[64, 64],
    batch_size=8,
    output_channels=1,
    alpha_ranges=[(0.8, 1.2), (0.8, 1.2)],
    uncorrelated_channel_prob=0.2,
)
# --8<-- [end:microsplit_synthetic_inputs]

# --8<-- [start:microsplit_loss]
config = create_advanced_microsplit_config(
    experiment_name="microsplit_loss",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    output_channels=1,
    reconstruction_weight=1.0,
    kl_weight=1.0,
    gaussian_likelihood_weight=0.1,
    noise_model_likelihood_weight=0.9,
    predict_logvar=True,
    logvar_lowerbound=-5.0,
)
# --8<-- [end:microsplit_loss]

# --8<-- [start:microsplit_model]
config = create_advanced_microsplit_config(
    experiment_name="microsplit_model",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    output_channels=1,
    model_params={
        "z_dims": [128, 128],
        "n_filters": 32,
        "encoder_dropout": 0.1,
        "decoder_dropout": 0.1,
    },
)
# --8<-- [end:microsplit_model]
