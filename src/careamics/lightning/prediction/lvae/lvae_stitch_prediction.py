"""Prediction utility functions."""

import numpy as np

from careamics.lvae_training.dataset.types import TilingMode


def stitch_prediction_vae(predictions, dset):
    """Stitch predictions back together using dataset's index manager.

    Parameters
    ----------
    predictions : numpy.ndarray
        Array of predictions with shape (n_tiles, channels, height, width).
    dset : Dataset
        Dataset object with idx_manager containing tiling information.

    Returns
    -------
    numpy.ndarray
        Stitched predictions.
    """
    mng = dset.idx_manager

    # if there are more channels, use all of them.
    shape = list(dset.get_data_shape())
    shape[-1] = max(shape[-1], predictions.shape[1])

    output = np.zeros(shape, dtype=predictions.dtype)
    # frame_shape = dset.get_data_shape()[:-1]
    for dset_idx in range(predictions.shape[0]):
        # loc = get_location_from_idx(dset, dset_idx, predictions.shape[-2],
        # predictions.shape[-1])
        # grid start, grid end
        gs = np.array(mng.get_location_from_dataset_idx(dset_idx), dtype=int)
        ge = gs + mng.grid_shape

        # patch start, patch end
        ps = gs - mng.patch_offset()
        pe = ps + mng.patch_shape

        # valid grid start, valid grid end
        vgs = np.array([max(0, x) for x in gs], dtype=int)
        vge = np.array(
            [min(x, y) for x, y in zip(ge, mng.data_shape, strict=False)], dtype=int
        )

        if mng.tiling_mode == TilingMode.ShiftBoundary:
            for dim in range(len(vgs)):
                if ps[dim] == 0:
                    vgs[dim] = 0
                if pe[dim] == mng.data_shape[dim]:
                    vge[dim] = mng.data_shape[dim]

        # relative start, relative end. This will be used on pred_tiled
        rs = vgs - ps
        re = rs + (vge - vgs)

        for ch_idx in range(predictions.shape[1]):
            if len(output.shape) == 4:
                # channel dimension is the last one.
                output[vgs[0] : vge[0], vgs[1] : vge[1], vgs[2] : vge[2], ch_idx] = (
                    predictions[dset_idx][ch_idx, rs[1] : re[1], rs[2] : re[2]]
                )
            elif len(output.shape) == 5:
                # channel dimension is the last one.
                assert vge[0] - vgs[0] == 1, "Only one frame is supported"
                output[
                    vgs[0], vgs[1] : vge[1], vgs[2] : vge[2], vgs[3] : vge[3], ch_idx
                ] = predictions[dset_idx][
                    ch_idx, rs[1] : re[1], rs[2] : re[2], rs[3] : re[3]
                ]
            else:
                raise ValueError(f"Unsupported shape {output.shape}")

    return output
