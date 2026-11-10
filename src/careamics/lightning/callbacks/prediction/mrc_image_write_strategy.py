"""Whole-image MRC writing strategy."""

from pathlib import Path
from typing import Any

import mrcfile
import numpy as np
from numpy.typing import NDArray

from careamics.lightning.prediction import combine_samples

from .file_path_utils import create_write_file_path, remove_even_odd_suffix
from .image_write_strategy import ImageWriteStrategy
from .mrc_tile_write_strategy import _OUTPUT_HEADER_FIELDS, _set_header_field


class MrcImageWriteStrategy(ImageWriteStrategy):
    """Write complete, non-tiled predictions to MRC files.

    This specialized strategy retains the source region metadata while collecting
    samples, then restores the transferable MRC header fields when the complete image
    is written.
    """

    def __init__(self) -> None:
        """Initialize the whole-image MRC writer."""
        super().__init__(
            write_func=lambda file_path, img, **kwargs: None,
            write_extension=".mrc",
            write_func_kwargs={},
        )

    def _write_mrc(
        self,
        file_path: Path,
        image: NDArray,
        metadata: dict[str, Any],
        axes: str,
    ) -> None:
        """Write an image and restore its transferable MRC metadata.

        Parameters
        ----------
        file_path : pathlib.Path
            Destination MRC path.
        image : numpy.ndarray
            Complete prediction image.
        metadata : dict of str to Any
            MRC header metadata from the source image.
        axes : str
            Axis order of ``image``.
        """
        if axes not in ("YX", "ZYX") or image.ndim != len(axes):
            raise ValueError(
                "MRC image writing requires a two- or three-dimensional YX/ZYX "
                f"image. Found axes '{axes}' and shape {image.shape}."
            )
        if file_path.exists():
            raise FileExistsError(f"File already exists at '{file_path}'.")

        with mrcfile.new(file_path, overwrite=True) as mrc:
            mrc.set_data(np.asarray(image))
            for name, value in metadata.items():
                if name not in _OUTPUT_HEADER_FIELDS:
                    _set_header_field(mrc.header, name, value)

    def _write_complete_images(self, dirpath: Path) -> None:
        """Write complete cached images as MRC files.

        Parameters
        ----------
        dirpath : pathlib.Path
            Directory where output files are written.
        """
        for data_idx in self._get_complete_images():
            cached_preds = self.image_cache.pop(data_idx)
            images, sources = combine_samples(cached_preds, restore_shape=True)
            metadata = cached_preds[0].additional_metadata.get("mrc", {})
            axes = cached_preds[0].target_axes

            for image, source in zip(images, sources, strict=True):
                # for cryoCARE, we want to remove "odd" or "even"
                source_path = remove_even_odd_suffix(Path(source))
                postfix = f"_{data_idx}" if source_path.stem == "array" else ""
                file_path = create_write_file_path(
                    dirpath=dirpath,
                    file_path=source_path,
                    write_extension=".mrc",
                    postfix=postfix,
                    source_base=self.source_base,
                )
                file_path.parent.mkdir(parents=True, exist_ok=True)
                self._write_mrc(file_path, image, metadata, axes)
