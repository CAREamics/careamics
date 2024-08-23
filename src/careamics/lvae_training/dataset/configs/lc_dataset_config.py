from pydantic import ConfigDict

from careamics.lvae_training.dataset.configs.vae_data_config import VaeDatasetConfig


class LCVaeDatasetConfig(VaeDatasetConfig):
    model_config = ConfigDict(validate_assignment=True)

    num_scales: int = 1
    """The number of resolutions at which we want the input. The target is formed at the
    highest resolution."""
