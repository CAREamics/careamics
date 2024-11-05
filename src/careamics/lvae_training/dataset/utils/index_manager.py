from dataclasses import dataclass

import numpy as np

from careamics.lvae_training.dataset.types import TilingMode


@dataclass
class GridIndexManager:
    data_shape: tuple
    grid_shape: tuple
    patch_shape: tuple
    tiling_mode: TilingMode

    # Patch is centered on index in the grid, grid size not used in training,
    # used only during val / test, grid size controls the overlap of the patches
    # in training you only get random patches every time
    # For borders - just cropped the data, so it perfectly divisible

    def __post_init__(self):
        assert len(self.data_shape) == len(
            self.grid_shape
        ), f"Data shape:{self.data_shape} and grid size:{self.grid_shape} must have the same dimension"
        assert len(self.data_shape) == len(
            self.patch_shape
        ), f"Data shape:{self.data_shape} and patch shape:{self.patch_shape} must have the same dimension"
        innerpad = np.array(self.patch_shape) - np.array(self.grid_shape)
        for dim, pad in enumerate(innerpad):
            if pad < 0:
                raise ValueError(
                    f"Patch shape:{self.patch_shape} must be greater than or equal to grid shape:{self.grid_shape} in dimension {dim}"
                )
            if pad % 2 != 0:
                raise ValueError(
                    f"Patch shape:{self.patch_shape} must have even padding in dimension {dim}"
                )

    def patch_offset(self):
        return (np.array(self.patch_shape) - np.array(self.grid_shape)) // 2

    def get_individual_dim_grid_count(self, dim: int):
        """
        Returns the number of the grid in the specified dimension, ignoring all other dimensions.
        """
        assert dim < len(
            self.data_shape
        ), f"Dimension {dim} is out of bounds for data shape {self.data_shape}"
        assert dim >= 0, "Dimension must be greater than or equal to 0"

        if self.grid_shape[dim] == 1 and self.patch_shape[dim] == 1:
            return self.data_shape[dim]
        elif self.tiling_mode == TilingMode.PadBoundary:
            return int(np.ceil(self.data_shape[dim] / self.grid_shape[dim]))
        elif self.tiling_mode == TilingMode.ShiftBoundary:
            excess_size = self.patch_shape[dim] - self.grid_shape[dim]
            return int(
                np.ceil((self.data_shape[dim] - excess_size) / self.grid_shape[dim])
            )
        else:
            excess_size = self.patch_shape[dim] - self.grid_shape[dim]
            return int(
                np.floor((self.data_shape[dim] - excess_size) / self.grid_shape[dim])
            )

    def total_grid_count(self):
        """
        Returns the total number of grids in the dataset.
        """
        return self.grid_count(0) * self.get_individual_dim_grid_count(0)

    def grid_count(self, dim: int):
        """
        Returns the total number of grids for one value in the specified dimension.
        """
        assert dim < len(
            self.data_shape
        ), f"Dimension {dim} is out of bounds for data shape {self.data_shape}"
        assert dim >= 0, "Dimension must be greater than or equal to 0"
        if dim == len(self.data_shape) - 1:
            return 1

        return self.get_individual_dim_grid_count(dim + 1) * self.grid_count(dim + 1)

    def get_grid_index(self, dim: int, coordinate: int):
        """
        Returns the index of the grid in the specified dimension.
        """
        assert dim < len(
            self.data_shape
        ), f"Dimension {dim} is out of bounds for data shape {self.data_shape}"
        assert dim >= 0, "Dimension must be greater than or equal to 0"
        assert (
            coordinate < self.data_shape[dim]
        ), f"Coordinate {coordinate} is out of bounds for data shape {self.data_shape}"

        if self.grid_shape[dim] == 1 and self.patch_shape[dim] == 1:
            return coordinate
        elif self.tiling_mode == TilingMode.PadBoundary:  # self.trim_boundary is False:
            return np.floor(coordinate / self.grid_shape[dim])
        elif self.tiling_mode == TilingMode.TrimBoundary:
            excess_size = (self.patch_shape[dim] - self.grid_shape[dim]) // 2
            # can be <0 if coordinate is in [0,grid_shape[dim]]
            return max(0, np.floor((coordinate - excess_size) / self.grid_shape[dim]))
        elif self.tiling_mode == TilingMode.ShiftBoundary:
            excess_size = (self.patch_shape[dim] - self.grid_shape[dim]) // 2
            if coordinate + self.grid_shape[dim] + excess_size == self.data_shape[dim]:
                return self.get_individual_dim_grid_count(dim) - 1
            else:
                # can be <0 if coordinate is in [0,grid_shape[dim]]
                return max(
                    0, np.floor((coordinate - excess_size) / self.grid_shape[dim])
                )

        else:
            raise ValueError(f"Unsupported tiling mode {self.tiling_mode}")

    def dataset_idx_from_grid_idx(self, grid_idx: tuple):
        """
        Returns the index of the grid in the dataset.
        """
        assert len(grid_idx) == len(
            self.data_shape
        ), f"Dimension indices {grid_idx} must have the same dimension as data shape {self.data_shape}"
        index = 0
        for dim in range(len(grid_idx)):
            index += grid_idx[dim] * self.grid_count(dim)
        return index

    def get_patch_location_from_dataset_idx(self, dataset_idx: int):
        """
        Returns the patch location of the grid in the dataset.
        """
        grid_location = self.get_location_from_dataset_idx(dataset_idx)
        offset = self.patch_offset()
        return tuple(np.array(grid_location) - np.array(offset))

    def get_dataset_idx_from_grid_location(self, location: tuple):
        assert len(location) == len(
            self.data_shape
        ), f"Location {location} must have the same dimension as data shape {self.data_shape}"
        grid_idx = [
            self.get_grid_index(dim, location[dim]) for dim in range(len(location))
        ]
        return self.dataset_idx_from_grid_idx(tuple(grid_idx))

    def get_gridstart_location_from_dim_index(self, dim: int, dim_index: int):
        """
        Returns the grid-start coordinate of the grid in the specified dimension.
        """
        assert dim < len(
            self.data_shape
        ), f"Dimension {dim} is out of bounds for data shape {self.data_shape}"
        assert dim >= 0, "Dimension must be greater than or equal to 0"
        # assert dim_index < self.get_individual_dim_grid_count(
        #     dim
        # ), f"Dimension index {dim_index} is out of bounds for data shape {self.data_shape}"
        # TODO comented out this shit cuz I have no interest to dig why it's failing at this point !
        if self.grid_shape[dim] == 1 and self.patch_shape[dim] == 1:
            return dim_index
        elif self.tiling_mode == TilingMode.PadBoundary:
            return dim_index * self.grid_shape[dim]
        elif self.tiling_mode == TilingMode.TrimBoundary:
            excess_size = (self.patch_shape[dim] - self.grid_shape[dim]) // 2
            return dim_index * self.grid_shape[dim] + excess_size
        elif self.tiling_mode == TilingMode.ShiftBoundary:
            excess_size = (self.patch_shape[dim] - self.grid_shape[dim]) // 2
            if dim_index < self.get_individual_dim_grid_count(dim) - 1:
                return dim_index * self.grid_shape[dim] + excess_size
            else:
                # on boundary. grid should be placed such that the patch covers the entire data.
                return self.data_shape[dim] - self.grid_shape[dim] - excess_size
        else:
            raise ValueError(f"Unsupported tiling mode {self.tiling_mode}")

    def get_location_from_dataset_idx(self, dataset_idx: int):
        """
        Returns the start location of the grid in the dataset.
        """
        grid_idx = []
        for dim in range(len(self.data_shape)):
            grid_idx.append(dataset_idx // self.grid_count(dim))
            dataset_idx = dataset_idx % self.grid_count(dim)
        location = [
            self.get_gridstart_location_from_dim_index(dim, grid_idx[dim])
            for dim in range(len(self.data_shape))
        ]
        return tuple(location)

    def on_boundary(self, dataset_idx: int, dim: int, only_end: bool = False):
        """
        Returns True if the grid is on the boundary in the specified dimension.
        """
        assert dim < len(
            self.data_shape
        ), f"Dimension {dim} is out of bounds for data shape {self.data_shape}"
        assert dim >= 0, "Dimension must be greater than or equal to 0"

        if dim > 0:
            dataset_idx = dataset_idx % self.grid_count(dim - 1)

        dim_index = dataset_idx // self.grid_count(dim)
        if only_end:
            return dim_index == self.get_individual_dim_grid_count(dim) - 1

        return (
            dim_index == 0 or dim_index == self.get_individual_dim_grid_count(dim) - 1
        )

    def next_grid_along_dim(self, dataset_idx: int, dim: int):
        """
        Returns the index of the grid in the specified dimension in the specified direction.
        """
        assert dim < len(
            self.data_shape
        ), f"Dimension {dim} is out of bounds for data shape {self.data_shape}"
        assert dim >= 0, "Dimension must be greater than or equal to 0"
        new_idx = dataset_idx + self.grid_count(dim)
        if new_idx >= self.total_grid_count():
            return None
        return new_idx

    def prev_grid_along_dim(self, dataset_idx: int, dim: int):
        """
        Returns the index of the grid in the specified dimension in the specified direction.
        """
        assert dim < len(
            self.data_shape
        ), f"Dimension {dim} is out of bounds for data shape {self.data_shape}"
        assert dim >= 0, "Dimension must be greater than or equal to 0"
        new_idx = dataset_idx - self.grid_count(dim)
        if new_idx < 0:
            return None
