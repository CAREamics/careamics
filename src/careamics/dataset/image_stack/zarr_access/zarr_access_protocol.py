"""Protocol for Zarr access backends."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from numpy.typing import DTypeLike, NDArray


@dataclass(frozen=True)
class ZarrNode:
    """Reference to a Zarr node.

    Attributes
    ----------
    store_uri : str
        URI of the backing Zarr store.
    path : str, default=""
        Path to the node within the store.
    node_type : {"array", "group"} or None, default=None
        Optional expected node type.
    """

    store_uri: str
    path: str = ""
    node_type: Literal["array", "group"] | None = None

    @property
    def parent_path(self) -> str:
        """Parent path within the Zarr store.

        Returns
        -------
        str
            Parent path relative to the store root.
        """
        if self.path == "":
            return ""
        return self.path.rpartition("/")[0]

    @property
    def basename(self) -> str:
        """Last path component within the Zarr store.

        Returns
        -------
        str
            Final path component.
        """
        if self.path == "":
            return ""
        return self.path.rpartition("/")[2]

    @property
    def source(self) -> str:
        """Normalized source string.

        Returns
        -------
        str
            Full store URI plus in-store path when present.
        """
        if self.path == "":
            return self.store_uri
        return f"{self.store_uri}/{self.path}"


@dataclass(frozen=True)
class ZarrArraySpec:
    """Specification for creating a Zarr array.

    Attributes
    ----------
    shape : tuple[int, ...]
        Array shape.
    chunks : tuple[int, ...]
        Chunk shape.
    shards : tuple[int, ...] or None
        Shard shape.
    dtype : DTypeLike
        Array dtype.
    dimension_names : tuple[str or None, ...] or None
        Optional dimension names.
    """

    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    shards: tuple[int, ...] | None
    dtype: DTypeLike
    dimension_names: tuple[str | None, ...] | None = None


class ZarrAccessProtocol(Protocol):
    """Protocol for backend-specific Zarr operations."""

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
        ...

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
        ...

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
        ...

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
        ...

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
        ...

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
        ...

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
        ...
