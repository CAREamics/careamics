import itertools
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union


def get_stratified_coords(num_pixels, shape):
    #TODO add description, add asserts, add typing
    box_size = np.round(np.sqrt(np.product(shape) / num_pixels)).astype(np.int32)
    box_count = [range(int(np.ceil(s / box_size))) for s in shape]
    output_coords = []
    for dim in itertools.product(*box_count):
        coords_random_increment = np.random.randint(0, box_size, size=len(shape))
        final_coords = [i * box_size for i in dim] + coords_random_increment
        if np.all(final_coords < shape):    
            output_coords.append(final_coords.tolist())
    return output_coords


def apply_struct_n2v_mask(patch, coords, dims, mask):
    """
    each point in coords corresponds to the center of the mask.
    then for point in the mask with value=1 we assign a random value
    """
    coords = np.array(coords).astype(np.int32)
    ndim = mask.ndim
    center = np.array(mask.shape) // 2
    ## leave the center value alone
    mask[tuple(center.T)] = 0j
    ## displacements from center
    dx = np.indices(mask.shape)[:, mask == 1] - center[:, None]
    ## combine all coords (ndim, npts,) with all displacements (ncoords,ndim,)
    mix = dx.T[..., None] + coords[None]
    mix = mix.transpose([1, 0, 2]).reshape([ndim, -1]).T
    ## stay within patch boundary
    mix = mix.clip(min=np.zeros(ndim), max=np.array(patch.shape) - 1).astype(np.uint)
    ## replace neighbouring pixels with random values from flat dist
    patch[tuple(mix.T)] = np.random.rand(mix.shape[0]) * 4 - 2
    #TODO finish, test
    return patch


def n2v_manipulate(patch: np.ndarray, num_pixels: int, augmentations: Callable = None) -> Tuple[np.ndarray, Dict]:
    """_summary_

    _extended_summary_

    Parameters
    ----------
    patch : _type_
        _description_
    num_pixels : _type_
        _description_
    """
    #
    #TODO check patch tuple ?
    # patch = patch[0]
    original_patch = patch.copy()
    mask = np.zeros(patch.shape)

    hot_pixels = get_stratified_coords(num_pixels, patch.shape)
    #TODO add another neighborhood selection strategy
    for pn in hot_pixels:
        #mf why you named it like this?
        rr = patch[(..., *[slice(max(c-2, 0), min(c+3, patch.shape[i])) for i, c in enumerate(pn)])]
        center_mask = np.isin(rr, patch[(..., *[c for c in pn])])
        try:
            replacement = np.random.default_rng().choice(np.delete(rr, center_mask.flatten()))
        except ValueError:
            #TODO check if replacement shape is correct, if center mask is correct
            replacement = np.random.default_rng().choice(rr.flatten()) + np.random.randint(-3, 3)
        patch[(..., *[c for c in pn])] = replacement
        mask[(..., *[c for c in pn])] = 1.0

    patch, mask = patch, mask if augmentations is None else augmentations(patch, mask)
    #TODO assert this output format ? 1st is required, others are optional
    return patch, original_patch, mask




