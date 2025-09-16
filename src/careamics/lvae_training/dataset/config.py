from typing import Any, Union

from pydantic import BaseModel, ConfigDict

from .types import DataSplitType, DataType, TilingMode


# TODO: check if any bool logic can be removed
class MicroSplitDataConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="allow")

    data_type: Union[DataType, str] | None  # TODO remove or refactor!!
    """Type of the dataset, should be one of DataType"""

    depth3D: int | None = 1
    """Number of slices in 3D. If data is 2D depth3D is equal to 1"""

    datasplit_type: DataSplitType | None = None
    """Whether to return training, validation or test split, should be one of
    DataSplitType"""

    num_channels: int | None = 2
    """Number of channels in the input"""

    # TODO: remove ch*_fname parameters, should be parsed automatically from a name list
    ch1_fname: str | None = None
    ch2_fname: str | None = None
    ch_input_fname: str | None = None

    input_is_sum: bool | None = False
    """Whether the input is the sum or average of channels"""

    input_idx: int | None = None
    """Index of the channel where the input is stored in the data"""

    target_idx_list: list[int] | None = None
    """Indices of the channels where the targets are stored in the data"""

    # TODO: where are there used?
    start_alpha: Any | None = None
    end_alpha: Any | None = None

    image_size: tuple  # TODO: revisit, new model_config uses tuple
    """Size of one patch of data"""

    grid_size: Union[int, tuple[int, int, int]] | None = None
    """Frame is divided into square grids of this size. A patch centered on a grid 
    having size `image_size` is returned. Grid size not used in training,
    used only during val / test, grid size controls the overlap of the patches"""

    empty_patch_replacement_enabled: bool | None = False
    """Whether to replace the content of one of the channels
    with background with given probability"""
    empty_patch_replacement_channel_idx: Any | None = None
    empty_patch_replacement_probab: Any | None = None
    empty_patch_max_val_threshold: Any | None = None

    uncorrelated_channels: bool | None = False
    """Replace the content in one of the channels with given probability to make
    channel content 'uncorrelated'"""
    uncorrelated_channel_probab: float | None = 0.5

    poisson_noise_factor: float | None = -1
    """The added poisson noise factor"""

    synthetic_gaussian_scale: float | None = 0.1

    # TODO: set to True in training code, recheck
    input_has_dependant_noise: bool | None = False

    # TODO: sometimes max_val differs between runs with fixed seeds with noise enabled
    enable_gaussian_noise: bool | None = False
    """Whether to enable gaussian noise"""

    # TODO: is this parameter used?
    allow_generation: bool = False

    # TODO: both used in IndexSwitcher, insure correct passing
    training_validtarget_fraction: Any = None
    deterministic_grid: Any = None

    # TODO: why is this not used?
    enable_rotation_aug: bool | None = False

    max_val: Union[float, tuple] | None = None
    """Maximum data in the dataset. Is calculated for train split, and should be
    externally set for val and test splits."""

    overlapping_padding_kwargs: Any = None
    """Parameters for np.pad method"""

    # TODO: remove this parameter, controls debug print
    print_vars: bool | None = False

    # Hard-coded parameters (used to be in the config file)
    normalized_input: bool = True
    """If this is set to true, then one mean and stdev is used
                for both channels. Otherwise, two different mean and stdev are used."""
    use_one_mu_std: bool | None = True

    # TODO: is this parameter used?
    train_aug_rotate: bool | None = False
    enable_random_cropping: bool | None = True

    multiscale_lowres_count: int | None = None
    """Number of LC scales"""

    tiling_mode: TilingMode | None = TilingMode.ShiftBoundary

    target_separate_normalization: bool | None = True

    mode_3D: bool | None = False
    """If training in 3D mode or not"""

    trainig_datausage_fraction: float | None = 1.0

    validtarget_random_fraction: float | None = None

    validation_datausage_fraction: float | None = 1.0

    random_flip_z_3D: bool | None = False

    padding_kwargs: dict = {"mode": "reflect"}  # TODO remove !!

    def __init__(self, **data):
        # Convert string data_type to enum if needed
        if "data_type" in data and isinstance(data["data_type"], str):
            try:
                data["data_type"] = DataType[data["data_type"]]
            except KeyError:
                # Keep original value to let validation handle the error
                pass
        super().__init__(**data)

    # TODO add validators !
