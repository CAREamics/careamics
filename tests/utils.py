from typing import Any

# -- Default values for testing

DEFAULT_PATCHING = "stratified"
DEFAULT_AXES = "YX"
DEFAULT_DATA_TYPE = "array"
DEFAULT_MODE = "training"


def patch_size_testing(data_type: str = DEFAULT_DATA_TYPE, axes: str = DEFAULT_AXES):
    """Return a patch size compatible with the data_type and axes."""
    if data_type == "czi" and axes == "SCTYX":
        return (8, 16, 16)
    elif "Z" in axes:
        return (8, 16, 16)
    else:
        return (16, 16)


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


def patch_filter_dict_testing(name: str = "shannon") -> dict[str, Any]:
    """Return a patch filter configuration dictionary."""
    match name:
        case "shannon":
            return {
                "name": name,
                "threshold": 0.5,
            }
        case "max":
            return {
                "name": name,
                "threshold": 0.5,
            }
        case "mean_std":
            return {
                "name": name,
                "mean_threshold": 0.5,
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
    # TODO: add normalization
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
