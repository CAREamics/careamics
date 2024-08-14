import numpy as np


class TilingManager:

    def __init__(
        self,
        data_shape: tuple[int, ...],
        tile_size: tuple[int, ...],
        overlaps: tuple[int, ...],
        trim_boundary: tuple[int, ...],
    ):
        # --- validation
        if len(data_shape) != len(tile_size):
            raise ValueError(
                f"Data shape:{data_shape} and tile size:{tile_size} must have the "
                "same dimension"
            )
        if len(data_shape) != len(overlaps):
            raise ValueError(
                f"Data shape:{data_shape} and tile overlaps:{overlaps} must have the "
                "same dimension"
            )
        # overlaps = np.array(tile_size) - np.array(grid_shape)
        if (np.array(overlaps) < 0).any():
            raise ValueError("Tile overlap must be positive or zero in all dimension.")
        if ((np.array(overlaps) % 2) != 0).any():
            # TODO: currently not required by CAREamics tiling,
            #   -> because floor divide is used.
            raise ValueError("Tile overlaps must be even.")

        # initialize attributes
        self.data_shape = data_shape
        self.grid_shape = tuple(np.array(tile_size) - np.array(overlaps))
        self.patch_shape = tile_size
        self.trim_boundary = trim_boundary

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
        elif self.trim_boundary is False:
            return int(np.ceil(self.data_shape[dim] / self.grid_shape[dim]))
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
        elif self.trim_boundary is False:
            return np.floor(coordinate / self.grid_shape[dim])
        else:
            excess_size = (self.patch_shape[dim] - self.grid_shape[dim]) // 2
            # can be <0 if coordinate is in [0,grid_shape[dim]]
            return max(0, np.floor((coordinate - excess_size) / self.grid_shape[dim]))

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
        location = self.get_location_from_dataset_idx(dataset_idx)
        offset = self.patch_offset()
        return tuple(np.array(location) - np.array(offset))

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
        assert dim_index < self.get_individual_dim_grid_count(
            dim
        ), f"Dimension index {dim_index} is out of bounds for data shape {self.data_shape}"

        if self.grid_shape[dim] == 1 and self.patch_shape[dim] == 1:
            return dim_index
        elif self.trim_boundary is False:
            return dim_index * self.grid_shape[dim]
        else:
            excess_size = (self.patch_shape[dim] - self.grid_shape[dim]) // 2
            return dim_index * self.grid_shape[dim] + excess_size

    def get_location_from_dataset_idx(self, dataset_idx: int):
        grid_idx = []
        for dim in range(len(self.data_shape)):
            grid_idx.append(dataset_idx // self.grid_count(dim))
            dataset_idx = dataset_idx % self.grid_count(dim)
        location = [
            self.get_gridstart_location_from_dim_index(dim, grid_idx[dim])
            for dim in range(len(self.data_shape))
        ]
        return tuple(location)

    def on_boundary(self, dataset_idx: int, dim: int):
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


if __name__ == "__main__":
    data_shape = (1, 1, 103, 103, 2)
    grid_shape = (1, 1, 16, 16, 2)
    patch_shape = (1, 1, 32, 32, 2)
    overlap = tuple(np.array(patch_shape) - np.array(grid_shape))

    trim_boundary = False
    manager = TilingManager(
        data_shape=data_shape,
        tile_size=patch_shape,
        overlaps=overlap,
        trim_boundary=trim_boundary,
    )
    gc = manager.total_grid_count()
    print("Grid count", gc)
    for i in range(gc):
        loc = manager.get_location_from_dataset_idx(i)
        print(i, loc)
        inferred_i = manager.get_dataset_idx_from_grid_location(loc)
        assert i == inferred_i, f"Index mismatch: {i} != {inferred_i}"

    for i in range(5):
        print(manager.on_boundary(40, i))
