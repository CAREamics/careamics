"""Configuration for UNet segmentation."""

from careamics.config.algorithms.seg_unet_algorithm_config import SegAlgorithm

from .ng_configuration import NGConfiguration


class SegConfiguration(NGConfiguration):
    """Configuration for segmentation tasks."""

    algorithm_config: SegAlgorithm
    """Algorithm configuration, holding all parameters required to configure the
    model."""
