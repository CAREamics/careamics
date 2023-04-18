import numpy as np


def calculate_number_of_patches(shape: int, patch_size: int) -> int:
    """Calculates the number of patches that can be extracted from an image of
    shape `shape` with patches of size `patch_size`.
    
    Parameters
    ----------
    shape : int
        Shape of the image
    patch_size : int
        Size of the patches
        
    Returns
    -------
    int
        Number of patches
    """
    return np.ceil(shape / patch_size).astype(int)


def calculate_overlap(shape: int, patch_size: int, num_patches: int) -> int:
    """Calculates the overlap between patches of size `patch_size` that can be
    extracted from an image of shape `shape` with `num_patches` patches.
    
    Parameters
    ----------
    shape : int
        Shape of the image
    patch_size : int
        Size of the patches
    num_patches : int
        Number of patches
        
    Returns
    -------
    int
        Overlap between patches
    """
    overlap = (num_patches * patch_size - shape) / max(1, num_patches - 1)

    return np.ceil(overlap).astype(int)
