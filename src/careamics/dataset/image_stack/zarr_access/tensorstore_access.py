"""TensorStore implementation of Zarr access."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import tensorstore as ts
from numpy import asarray
from numpy import dtype as np_dtype
from numpy.typing import DTypeLike, NDArray
from pydantic_tensorstore import (
    FileKvStore,
    Zarr3BytesConfig,
    Zarr3ChunkConfiguration,
    Zarr3ChunkGrid,
    Zarr3CodecBytes,
    Zarr3CodecCRC32C,
    Zarr3CodecShardingIndexed,
    Zarr3Metadata,
    Zarr3ShardingIndexedConfig,
    Zarr3Spec,
)

from .zarr_access_protocol import ZarrArraySpec, ZarrNode
from .zarr_access_utils import file_uri_to_path


class TensorstoreAccess:
    """TensorStore-backed Zarr access."""

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
        return self._require_array(node).dtype.numpy_dtype

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
        return tuple(self._require_array(node).chunk_layout.read_chunk.shape)

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
        array = self._require_array(node)
        metadata = array.spec().to_json().get("metadata", {})
        codecs = metadata.get("codecs", [])
        if any(codec.get("name") == "sharding_indexed" for codec in codecs):
            return tuple(array.chunk_layout.write_chunk.shape)
        return None

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
        return asarray(self._require_array(node, mode="r")[patch_index].read().result())

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
        store_path = file_uri_to_path(node.store_uri)
        if store_path.exists():
            try:
                self._open_tensorstore_array(node, mode="a")
                return
            except ValueError:
                pass

        # note: when sharding, shards are chunks with a specific codec and chunks
        # become inner chunks, see TensorstoreAccess._codecs
        metadata = Zarr3Metadata(
            shape=list(spec.shape),
            data_type=str(np_dtype(spec.dtype).name),
            chunk_grid=Zarr3ChunkGrid(
                configuration=Zarr3ChunkConfiguration(
                    chunk_shape=list(
                        spec.shards if spec.shards is not None else spec.chunks
                    )
                )
            ),
            codecs=self._codecs(spec.chunks) if spec.shards is not None else None,
            dimension_names=(
                list(spec.dimension_names) if spec.dimension_names is not None else None
            ),
        )
        tensorstore_spec = self._spec(node, metadata=metadata, create=True)
        ts.open(tensorstore_spec).result()

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
        self._require_array(node, mode="a")[tile_index].write(data).result()

    def _require_array(
        self, node: ZarrNode, mode: Literal["r", "a", "w"] = "r"
    ) -> ts.TensorStore:
        """Open a node as a TensorStore array.

        Parameters
        ----------
        node : ZarrNode
            Array node to open.
        mode : {"r", "a", "w"}, default="r"
            Open mode.

        Returns
        -------
        tensorstore.TensorStore
            Opened array.
        """
        opened = self._open_tensorstore_array(node, mode=mode)
        if not isinstance(opened, ts.TensorStore):
            raise TypeError(f"Node '{node.source}' is not a tensorstore.TensorStore.")
        return opened

    def _open_tensorstore_array(
        self, node: ZarrNode, mode: Literal["r", "a", "w"] = "r"
    ) -> ts.TensorStore:
        """Open a TensorStore array.

        Parameters
        ----------
        node : ZarrNode
            Array node to open.
        mode : {"r", "a", "w"}, default="r"
            Open mode.

        Returns
        -------
        tensorstore.TensorStore
            Opened array.
        """
        spec = self._spec(node)
        open_mode = mode in {"r", "a"}
        create_mode = mode in {"a", "w"}
        try:
            return ts.open(spec, open=open_mode, create=create_mode).result()
        except Exception as error:
            raise ValueError(f"Could not open Zarr array '{node.source}'.") from error

    def _spec(
        self,
        node: ZarrNode,
        *,
        metadata: Zarr3Metadata | None = None,
        create: bool = False,
    ) -> dict[str, Any]:
        """Build a TensorStore specification.

        Parameters
        ----------
        node : ZarrNode
            Array node.
        metadata : Zarr3Metadata or None, default=None
            Optional array metadata.
        create : bool, default=False
            Whether the specification creates an array.

        Returns
        -------
        dict[str, Any]
            Serialized TensorStore specification.
        """
        store_path = file_uri_to_path(node.store_uri)
        spec = Zarr3Spec(
            kvstore=FileKvStore(path=str(store_path)),
            path=node.path,
            metadata=metadata,
            create=create,
        )
        return spec.model_dump(by_alias=True, exclude_none=True)

    def _codecs(self, chunks: tuple[int, ...]) -> list[Zarr3CodecShardingIndexed]:
        """Build a sharding codec.

        Parameters
        ----------
        chunks : tuple[int, ...]
            Inner chunk shape.

        Returns
        -------
        list[Zarr3CodecShardingIndexed]
            Sharding codec chain.
        """
        return [
            Zarr3CodecShardingIndexed(
                configuration=Zarr3ShardingIndexedConfig(
                    chunk_shape=list(chunks),
                    codecs=[
                        Zarr3CodecBytes(configuration=Zarr3BytesConfig(endian="little"))
                    ],
                    index_codecs=[
                        Zarr3CodecBytes(
                            configuration=Zarr3BytesConfig(endian="little")
                        ),
                        Zarr3CodecCRC32C(),
                    ],
                    index_location="end",
                )
            )
        ]
