"""`zarr` implementation of Zarr access."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import zarr
from numpy import asarray
from numpy.typing import DTypeLike, NDArray

from .zarr_access_protocol import ZarrArraySpec, ZarrNode
from .zarr_access_utils import file_uri_to_path
from .zarr_hierarchy_utils import open_zarr_node


class ZarrPythonAccess:
    """`zarr`-backed Zarr access."""

    def get_array_shape(self, node: ZarrNode) -> tuple[int, ...]:
        """Return an array shape.

        Parameters
        ----------
        node : ZarrNode
            Array node to inspect.

        Returns
        -------
        tuple[int, ...]
            Array shape.
        """
        return tuple(self._require_array(node).shape)

    def get_array_dtype(self, node: ZarrNode) -> DTypeLike:
        """Return an array dtype.

        Parameters
        ----------
        node : ZarrNode
            Array node to inspect.

        Returns
        -------
        DTypeLike
            Array dtype.
        """
        return self._require_array(node).dtype

    def get_array_chunks(self, node: ZarrNode) -> Sequence[int]:
        """Return an array chunk shape.

        Parameters
        ----------
        node : ZarrNode
            Array node to inspect.

        Returns
        -------
        Sequence[int]
            Chunk shape.
        """
        return self._require_array(node).chunks

    def get_array_shards(self, node: ZarrNode) -> Sequence[int] | None:
        """Return an array shard shape.

        Parameters
        ----------
        node : ZarrNode
            Array node to inspect.

        Returns
        -------
        Sequence[int] or None
            Shard shape.
        """
        return self._require_array(node).shards

    def read_array_patch(self, node: ZarrNode, patch_index: Any) -> NDArray[Any]:
        """Read a patch from an array node.

        Parameters
        ----------
        node : ZarrNode
            Array node to read from.
        patch_index : Any
            Indexing object selecting the patch to read.

        Returns
        -------
        NDArray[Any]
            Selected patch data.
        """
        return asarray(self._require_array(node, mode="r")[patch_index])

    def create_array(
        self,
        node: ZarrNode,
        spec: ZarrArraySpec,
    ) -> None:
        """Create or open an output array.

        Parameters
        ----------
        node : ZarrNode
            Output array node.
        spec : ZarrArraySpec
            Output array specification. Parent groups must already exist.

        Returns
        -------
        None
            The array is created or opened in place.
        """
        if node.path == "":
            self._create_or_open_root_array(node, spec)
        else:
            self._create_or_open_group_array(node, spec)

    def write_array_tile(
        self, node: ZarrNode, tile_index: Any, data: NDArray[Any]
    ) -> None:
        """Write a tile to an array node.

        Parameters
        ----------
        node : ZarrNode
            Array node to write to.
        tile_index : Any
            Indexing object selecting the tile destination.
        data : NDArray[Any]
            Data to write.

        Returns
        -------
        None
            This method writes in place.
        """
        array = self._require_array(node, mode="a")
        array[tile_index] = data

    def _require_array(
        self, node: ZarrNode, mode: Literal["r", "a", "w"] = "r"
    ) -> zarr.Array:
        """Open a node and require that it is an array.

        Parameters
        ----------
        node : ZarrNode
            Node to inspect.
        mode : {"r", "a", "w"}, default="r"
            Open mode.

        Returns
        -------
        zarr.Array
            Opened array node.
        """
        opened = open_zarr_node(node, mode=mode)
        if not isinstance(opened, zarr.Array):
            raise TypeError(f"Node '{node.source}' is not a zarr.Array.")
        return opened

    def _create_or_open_root_array(
        self,
        node: ZarrNode,
        spec: ZarrArraySpec,
    ) -> zarr.Array:
        """Create or open a root-array output.

        Parameters
        ----------
        node : ZarrNode
            Root output array node.
        spec : ZarrArraySpec
            Output array specification.

        Returns
        -------
        zarr.Array
            Existing or newly created root array.
        """
        store_path = file_uri_to_path(node.store_uri)

        if store_path.exists():
            opened = zarr.open(store_path, mode="a")
            if not isinstance(opened, zarr.Array):
                raise RuntimeError(f"Zarr store at {store_path} is not a root array.")
            return opened

        opened = zarr.create_array(
            store=store_path,
            shape=spec.shape,
            chunks=spec.chunks,
            shards=spec.shards,
            dtype=spec.dtype,
            dimension_names=spec.dimension_names,
            zarr_format=3,
        )
        if not isinstance(opened, zarr.Array):
            raise RuntimeError(f"Zarr store at {store_path} is not a root array.")
        return opened

    def _create_or_open_group_array(
        self,
        node: ZarrNode,
        spec: ZarrArraySpec,
    ) -> zarr.Array:
        """Create or open a group-backed output array.

        Parameters
        ----------
        node : ZarrNode
            Output array node.
        spec : ZarrArraySpec
            Output array specification. Parent groups must already exist.

        Returns
        -------
        zarr.Array
            Existing or newly created group-backed array.
        """
        store_path = file_uri_to_path(node.store_uri)

        opened = zarr.open(store_path, mode="a")
        if not isinstance(opened, zarr.Group):
            raise RuntimeError(f"Zarr store at {store_path} is not a group.")

        group = opened
        if node.parent_path != "":
            existing_group = opened[node.parent_path]
            if not isinstance(existing_group, zarr.Group):
                raise RuntimeError(f"Zarr group at {node.parent_path} is not a group.")
            group = existing_group

        if node.basename not in group:
            array = group.create_array(
                name=node.basename,
                shape=spec.shape,
                shards=spec.shards,
                chunks=spec.chunks,
                dtype=spec.dtype,
                dimension_names=spec.dimension_names,
            )
        else:
            existing_array = group[node.basename]
            if not isinstance(existing_array, zarr.Array):
                raise RuntimeError(f"Zarr array at {node.path} is not an array.")
            array = existing_array

        return array
