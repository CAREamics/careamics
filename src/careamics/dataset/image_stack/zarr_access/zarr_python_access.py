"""`zarr` implementation of Zarr access."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import zarr
from numpy import asarray
from numpy.typing import DTypeLike, NDArray

from .zarr_access_protocol import ZarrNode
from .zarr_access_utils import file_uri_to_path


class ZarrPythonAccess:
    """`zarr`-backed Zarr access."""

    def open_node(
        self, node: ZarrNode, mode: Literal["r", "a", "w"] = "r"
    ) -> zarr.Array | zarr.Group:
        """Open a Zarr node.

        Parameters
        ----------
        node : ZarrNode
            Node to open.
        mode : {"r", "a", "w"}, default="r"
            Open mode.

        Returns
        -------
        zarr.Array or zarr.Group
            Opened node.
        """
        store_path = file_uri_to_path(node.store_uri)
        opened = zarr.open(store_path, mode=mode)

        if node.path == "":
            return opened

        if not isinstance(opened, zarr.Group):
            raise TypeError(
                f"Zarr store at '{store_path}' is an array, cannot access child path "
                f"'{node.path}'."
            )

        return opened[node.path]

    def list_array_paths(self, node: ZarrNode) -> list[str]:
        """List first-level arrays beneath a group node.

        Parameters
        ----------
        node : ZarrNode
            Group node to inspect.

        Returns
        -------
        list[str]
            Relative array paths.
        """
        opened = self.open_node(node, mode="r")
        if not isinstance(opened, zarr.Group):
            raise TypeError(f"Node '{node.source}' is not a zarr.Group.")

        return [
            name for name in opened.array_keys() if isinstance(opened[name], zarr.Array)
        ]

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

    def resolve_node_type(self, node: ZarrNode) -> ZarrNode:
        """Return a node with updated node type.

        Parameters
        ----------
        node : ZarrNode
            Node to inspect.

        Returns
        -------
        ZarrNode
            ZarrNode with updated type.

        Raises
        ------
        ValueError
            If the ZarrNode corresponds to neither zarr.Array nor zarr.Group.
        """
        opened = self.open_node(node, mode="r")
        if isinstance(opened, zarr.Array):
            node_type: Literal["array", "group"] = "array"
        elif isinstance(opened, zarr.Group):
            node_type = "group"
        else:
            raise ValueError(
                f"Unsupported Zarr node type for '{node.source}': {type(opened)}."
            )

        return ZarrNode(store_uri=node.store_uri, path=node.path, node_type=node_type)

    def create_array(
        self,
        node: ZarrNode,
        shape: Sequence[int],
        chunks: tuple[int, ...],
        shards: tuple[int, ...] | None,
        dtype: DTypeLike,
    ) -> zarr.Array:
        """Create or open an output array.

        Parameters
        ----------
        node : ZarrNode
            Output array node.
        shape : Sequence[int]
            Output array shape.
        chunks : tuple[int, ...]
            Output chunk shape.
        shards : tuple[int, ...] or None
            Output shard shape.
        dtype : DTypeLike
            Output array dtype.

        Returns
        -------
        zarr.Array
            Existing or newly created output array.
        """
        if node.path == "":
            return self._create_or_open_root_array(node, shape, chunks, shards, dtype)
        return self._create_or_open_group_array(node, shape, chunks, shards, dtype)

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
        opened = self.open_node(node, mode=mode)
        if not isinstance(opened, zarr.Array):
            raise TypeError(f"Node '{node.source}' is not a zarr.Array.")
        return opened

    def _create_or_open_root_array(
        self,
        node: ZarrNode,
        shape: Sequence[int],
        chunks: tuple[int, ...],
        shards: tuple[int, ...] | None,
        dtype: DTypeLike,
    ) -> zarr.Array:
        """Create or open a root-array output.

        Parameters
        ----------
        node : ZarrNode
            Root output array node.
        shape : Sequence[int]
            Output array shape.
        chunks : tuple[int, ...]
            Output chunk shape.
        shards : tuple[int, ...] or None
            Output shard shape.
        dtype : DTypeLike
            Output array dtype.

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

        if shards is not None:
            raise NotImplementedError(
                "Writing sharded root Zarr arrays is not supported with the current "
                "`zarr` backend. Please write to a group-backed array instead."
            )

        opened = zarr.open(
            store_path,
            mode="w",
            shape=shape,
            chunks=chunks,
            dtype=dtype,
        )
        if not isinstance(opened, zarr.Array):
            raise RuntimeError(f"Zarr store at {store_path} is not a root array.")
        return opened

    def _create_or_open_group_array(
        self,
        node: ZarrNode,
        shape: Sequence[int],
        chunks: tuple[int, ...],
        shards: tuple[int, ...] | None,
        dtype: DTypeLike,
    ) -> zarr.Array:
        """Create or open a group-backed output array.

        Parameters
        ----------
        node : ZarrNode
            Output array node.
        shape : Sequence[int]
            Output array shape.
        chunks : tuple[int, ...]
            Output chunk shape.
        shards : tuple[int, ...] or None
            Output shard shape.
        dtype : DTypeLike
            Output array dtype.

        Returns
        -------
        zarr.Array
            Existing or newly created group-backed array.
        """
        store_path = file_uri_to_path(node.store_uri)

        if store_path.exists():
            opened = zarr.open(store_path, mode="a")
            if not isinstance(opened, zarr.Group):
                raise RuntimeError(f"Zarr store at {store_path} is not a group.")
            store = opened
        else:
            store = zarr.create_group(store_path)

        group = store
        if node.parent_path != "":
            if node.parent_path not in store:
                group = store.create_group(node.parent_path)
            else:
                existing_group = store[node.parent_path]
                if not isinstance(existing_group, zarr.Group):
                    raise RuntimeError(
                        f"Zarr group at {node.parent_path} is not a group."
                    )
                group = existing_group

        if node.basename not in group:
            array = group.create_array(
                name=node.basename,
                shape=shape,
                shards=shards,
                chunks=chunks,
                dtype=dtype,
            )
        else:
            existing_array = group[node.basename]
            if not isinstance(existing_array, zarr.Array):
                raise RuntimeError(f"Zarr array at {node.path} is not an array.")
            array = existing_array

        return array
