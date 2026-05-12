"""
Demo the block validation selection feature.
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "careamics",
# ]
#
# [tool.uv.sources]
# careamics = { path = ".." }
# ///

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

from careamics.dataset.patching import Patching, StratifiedPatching
from careamics.dataset.factory.val_split import select_validation


def demo_selected_patches(
    patching_strategy: Patching,
    data_shapes: Sequence[Sequence[int]],
    epochs: int,
) -> Sequence[NDArray[np.int_]]:
    """Create a map where all the patches have been selected from.

    Every time a patch is selected that area is incremented by 1.

    Parameters
    ----------
    patching_strategy : Patching
        Patching strategy.
    data_shapes : Sequence[Sequence[int]]
        The shapes of the data. SC(Z)YX.
    epochs : int
        Number of epochs.

    Returns
    -------
    Sequence[NDArray[np.int_]]
        List of tracking arrays.
    """
    tracking_arrays = [np.zeros(shape, dtype=int) for shape in data_shapes]
    for _ in range(epochs):
        for index in range(patching_strategy.n_patches):
            patch_spec = patching_strategy.get_patch_spec(index)
            data_idx = patch_spec["data_idx"]
            sample_idx = patch_spec["sample_idx"]
            coord = patch_spec["coords"]
            patch_size = patch_spec["patch_size"]

            patch_slice = [
                slice(c, c + ps) for c, ps in zip(coord, patch_size, strict=True)
            ]
            tracking_arrays[data_idx][sample_idx, ..., *patch_slice] += 1
    return tracking_arrays


def plot_block_validation():
    """Show block selection for different image sizes and number of patches."""
    data_shapes = [(512, 512), (768, 768), (1024, 1024)]
    patch_size = (64, 64)
    fig, axes = plt.subplots(3, 4, figsize=(16, 12), constrained_layout=True)
    fig.suptitle("Block validation selection")
    rng = np.random.default_rng(seed=42)
    for i, data_shape in enumerate(data_shapes):
        axes[i][0].set_ylabel(f"Shape: {data_shape}")
        for j in range(4):
            stratified_patching = StratifiedPatching([(1, 1, *data_shape)], patch_size)
            if j == 0:
                n_val_patches = 8
                axes[0][j].set_title("N Val Patches: 8")
            else:
                n_val_patches = int(np.floor(stratified_patching.n_patches * (j * 0.1)))
                axes[0][j].set_title(f"N Val Patches: {j}0%")

            coords = select_validation(
                stratified_patching.image_patching[0][0].grid_shape,
                n_val_patches,
                rng,
            )
            stratified_patching.exclude_patches(0, 0, coords)

            tracking = demo_selected_patches(
                stratified_patching, stratified_patching.data_shapes, 50
            )
            axes[i][j].imshow(tracking[0][0, 0])


def plot_random_validation():
    """Show random selection for different image sizes and number of patches."""
    data_shapes = [(512, 512), (768, 768), (1024, 1024)]
    patch_size = (64, 64)
    fig, axes = plt.subplots(3, 4, figsize=(16, 12), constrained_layout=True)
    fig.suptitle("Random validation selection")
    rng = np.random.default_rng(seed=42)
    for i, data_shape in enumerate(data_shapes):
        axes[i][0].set_ylabel(f"Shape: {data_shape}")
        for j in range(4):
            stratified_patching = StratifiedPatching([(1, 1, *data_shape)], patch_size)
            if j == 0:
                n_val_patches = 8
                axes[0][j].set_title("N Val Patches: 8")
            else:
                n_val_patches = int(np.floor(stratified_patching.n_patches * (j * 0.1)))
                axes[0][j].set_title(f"N Val Patches: {j}0%")

            grid_coords = stratified_patching.get_included_grid_coords()[0, 0]
            # randomly choose the validation patches in the image
            coord_indices = rng.choice(len(grid_coords), n_val_patches, replace=False)
            coords: list[tuple[int, ...]] = [
                grid_coords[coord_idx] for coord_idx in coord_indices
            ]
            stratified_patching.exclude_patches(0, 0, coords)

            tracking = demo_selected_patches(
                stratified_patching, stratified_patching.data_shapes, 50
            )
            axes[i][j].imshow(tracking[0][0, 0])


def main():
    """Main."""
    plot_block_validation()
    plot_random_validation()
    plt.show()


if __name__ == "__main__":
    main()
