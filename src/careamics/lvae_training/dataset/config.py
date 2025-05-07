from typing import Any, Optional, Union

from pydantic import BaseModel, ConfigDict

from .types import DataSplitType, DataType, TilingMode


# TODO: check if any bool logic can be removed
class DatasetConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    data_type: Optional[DataType]
    """Type of the dataset, should be one of DataType"""

    depth3D: Optional[int] = 1
    """Number of slices in 3D. If data is 2D depth3D is equal to 1"""

    datasplit_type: Optional[DataSplitType] = None
    """Whether to return training, validation or test split, should be one of 
    DataSplitType"""

    num_channels: Optional[int] = 2
    """Number of channels in the input"""

    # TODO: remove ch*_fname parameters, should be parsed automatically from a name list
    ch1_fname: Optional[str] = None
    ch2_fname: Optional[str] = None
    ch_input_fname: Optional[str] = None

    input_is_sum: Optional[bool] = False
    """Whether the input is the sum or average of channels"""

    input_idx: Optional[int] = None
    """Index of the channel where the input is stored in the data"""

    target_idx_list: Optional[list[int]] = None
    """Indices of the channels where the targets are stored in the data"""

    # TODO: where are there used?
    start_alpha: Optional[Any] = None
    end_alpha: Optional[Any] = None

    image_size: tuple  # TODO: revisit, new model_config uses tuple
    """Size of one patch of data"""

    grid_size: Optional[Union[int, tuple[int, int, int]]] = None
    """Frame is divided into square grids of this size. A patch centered on a grid 
    having size `image_size` is returned. Grid size not used in training,
    used only during val / test, grid size controls the overlap of the patches"""

    empty_patch_replacement_enabled: Optional[bool] = False
    """Whether to replace the content of one of the channels
    with background with given probability"""
    empty_patch_replacement_channel_idx: Optional[Any] = None
    empty_patch_replacement_probab: Optional[Any] = None
    empty_patch_max_val_threshold: Optional[Any] = None

    uncorrelated_channels: Optional[bool] = False
    """Replace the content in one of the channels with given probability to make 
    channel content 'uncorrelated'"""
    uncorrelated_channel_probab: Optional[float] = 0.5

    poisson_noise_factor: Optional[float] = -1
    """The added poisson noise factor"""

    synthetic_gaussian_scale: Optional[float] = 0.1

    # TODO: set to True in training code, recheck
    input_has_dependant_noise: Optional[bool] = False

    # TODO: sometimes max_val differs between runs with fixed seeds with noise enabled
    enable_gaussian_noise: Optional[bool] = False
    """Whether to enable gaussian noise"""

    # TODO: is this parameter used?
    allow_generation: bool = False

    # TODO: both used in IndexSwitcher, insure correct passing
    training_validtarget_fraction: Any = None
    deterministic_grid: Any = None

    # TODO: why is this not used?
    enable_rotation_aug: Optional[bool] = False

    max_val: Optional[Union[float, tuple]] = None
    """Maximum data in the dataset. Is calculated for train split, and should be 
    externally set for val and test splits."""

    overlapping_padding_kwargs: Any = None
    """Parameters for np.pad method"""

    # TODO: remove this parameter, controls debug print
    print_vars: Optional[bool] = False

    # Hard-coded parameters (used to be in the config file)
    normalized_input: bool = True
    """If this is set to true, then one mean and stdev is used
                for both channels. Otherwise, two different mean and stdev are used."""
    use_one_mu_std: Optional[bool] = True

    # TODO: is this parameter used?
    train_aug_rotate: Optional[bool] = False
    enable_random_cropping: Optional[bool] = True

    multiscale_lowres_count: Optional[int] = None
    """Number of LC scales"""

    tiling_mode: Optional[TilingMode] = TilingMode.ShiftBoundary

    target_separate_normalization: Optional[bool] = True

    mode_3D: Optional[bool] = False
    """If training in 3D mode or not"""

    trainig_datausage_fraction: Optional[float] = 1.0

    validtarget_random_fraction: Optional[float] = None

    validation_datausage_fraction: Optional[float] = 1.0

    random_flip_z_3D: Optional[bool] = False

    padding_kwargs: Optional[dict] = None
