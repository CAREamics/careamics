from pydantic import ConfigDict

from careamics.lvae_training.dataset.configs.multich_data_config import (
    MultiChDatasetConfig,
)


class LCDatasetConfig(MultiChDatasetConfig):
    model_config = ConfigDict(validate_assignment=True)

    num_scales: int = 1
    """The number of resolutions at which we want the input. The target is formed at the
    highest resolution."""
