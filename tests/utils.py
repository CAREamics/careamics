from typing import Any

# -- Default values for testing

DEFAULT_PATCHING = "stratified"
DEFAULT_AXES = "YX"
DEFAULT_DATA_TYPE = "array"
DEFAULT_MODE = "training"
DEFAULT_DIMS = 2
DEFAULT_N_CHANNELS_IN = 1
DEFAULT_N_CHANNELS_OUT = 1
DEFAULT_ALGORITHM = "n2v"


def patch_size_testing(data_type: str = DEFAULT_DATA_TYPE, axes: str = DEFAULT_AXES):
    """Return a patch size compatible with the data_type and axes."""
    if data_type == "czi" and axes == "SCTYX":
        return (8, 32, 32)
    elif "Z" in axes:
        return (8, 32, 32)
    else:
        return (32, 32)


def patching_config_dict_testing(
    patching: str = DEFAULT_PATCHING,
    data_type: str = DEFAULT_DATA_TYPE,
    axes: str = DEFAULT_AXES,
    patch_size: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Return a patching configuration dictionary."""
    if patch_size is None:
        patch_size = patch_size_testing(data_type, axes)
    match patching:
        case "random" | "fixed_random" | "stratified":
            # with seed
            return {
                "name": patching,
                "patch_size": patch_size,
                "seed": 42,
            }
        case "tiled":
            # with overlaps
            return {
                "name": patching,
                "patch_size": patch_size,
                "overlaps": tuple(ps // 2 for ps in patch_size),
            }
        case _:
            return {
                "name": patching,
                "patch_size": patch_size,
            }


def patch_filter_dict_testing(
    name: str = "shannon", filtered_patch_prob: float = 0.1
) -> dict[str, Any]:
    """Return a patch filter configuration dictionary."""
    match name:
        case "shannon":
            return {
                "name": name,
                "threshold": 0.5,
                "filtered_patch_prob": filtered_patch_prob,
            }
        case "max":
            return {
                "name": name,
                "threshold": 0.5,
                "filtered_patch_prob": filtered_patch_prob,
            }
        case "mean_std":
            return {
                "name": name,
                "mean_threshold": 0.5,
                "filtered_patch_prob": filtered_patch_prob,
            }
        case _:
            raise ValueError(f"Invalid patch filter name: {name}")


def ng_data_config_dict_testing(
    mode: str = DEFAULT_MODE,
    data_type: str = DEFAULT_DATA_TYPE,
    axes: str = DEFAULT_AXES,
    patching: str | None = None,
    patch_size: tuple[int, ...] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Return a NGDataConfig dictionary."""
    if patching is None:
        match mode:
            case "training":
                patching = "stratified"
            case "validating":
                patching = "fixed_random"
            case "predicting":
                patching = "whole"
            case _:
                raise ValueError(f"Invalid mode: {mode}")

    patching_config_dict = patching_config_dict_testing(
        patching, data_type, axes, patch_size
    )
    return {
        "mode": mode,
        "data_type": data_type,
        "axes": axes,
        "patching": patching_config_dict,
        "normalization": {"name": "mean_std"},
        **kwargs,
    }


def unet_ng_algo_dict_testing(
    # algorithm
    algorithm: str = DEFAULT_ALGORITHM,
    dims: int = DEFAULT_DIMS,
    n_channels_in: int = DEFAULT_N_CHANNELS_IN,
    n_channels_out: int = DEFAULT_N_CHANNELS_OUT,
    **model_kwargs: Any,
) -> dict[str, Any]:
    """Return a UNet algorithm dictionary."""
    if algorithm == "n2v":
        loss = "n2v"
    else:
        loss = "mae"

    ind_channels = n_channels_in == n_channels_out

    return {
        "algorithm": algorithm,
        "loss": loss,
        "model": {
            "architecture": "UNet",
            "conv_dims": dims,
            "in_channels": n_channels_in,
            "num_classes": n_channels_out,
            "independent_channels": ind_channels,
            **model_kwargs,
        },
    }


# TODO not compatible with n2v2, structn2v, needs extension to N2VManipulateConfig
def unet_ng_config_dict_testing(
    # algorithm
    algorithm: str = DEFAULT_ALGORITHM,
    n_channels_in: int = DEFAULT_N_CHANNELS_IN,
    n_channels_out: int = DEFAULT_N_CHANNELS_OUT,
    model_kwargs: dict[str, Any] | None = None,
    # data
    mode: str = DEFAULT_MODE,
    data_type: str = DEFAULT_DATA_TYPE,
    axes: str = DEFAULT_AXES,
    patching: str | None = None,
    patch_size: tuple[int, ...] | None = None,
    data_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a NGConfiguration dictionary."""
    if patch_size is None:
        patch_size = patch_size_testing(data_type, axes)

    dims = len(patch_size)

    return {
        "experiment_name": "test_experiment",
        "algorithm_config": unet_ng_algo_dict_testing(
            algorithm, dims, n_channels_in, n_channels_out, **(model_kwargs or {})
        ),
        "data_config": ng_data_config_dict_testing(
            mode, data_type, axes, patching, patch_size, **(data_kwargs or {})
        ),
    }
