from collections.abc import Sequence


# TODO: move to dataset_utils, better name?
def reshaped_array_shape(axes: str, shape: Sequence[int]) -> tuple[int, ...]:
    """Find resulting shape if reshaping array with given `axes` and `shape`."""
    target_axes = "SCZYX"
    target_shape = []
    for d in target_axes:
        if d in axes:
            idx = axes.index(d)
            target_shape.append(shape[idx])
        elif (d != axes) and (d != "Z"):
            target_shape.append(1)
        else:
            pass

    if "T" in axes:
        idx = axes.index("T")
        target_shape[0] = target_shape[0] * shape[idx]

    return tuple(target_shape)
