from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

try:
    from pylibCZIrw.czi import CziReader, Rectangle, open_czi

    pyczi_available = True
except ImportError:
    pyczi_available = False

if TYPE_CHECKING:
    try:
        from pylibCZIrw.czi import CziReader, Rectangle, open_czi
    except ImportError:
        CziReader = Rectangle = open_czi = None  # type: ignore


class CziImageStack:
    """
    A class for extracting patches from an image stack that is stored as a CZI file.

    Parameters
    ----------
    data_path : str or Path
        Path to the CZI file.

    scene : int, optional
        Index of the scene to extract.

        A single CZI file can contain multiple "scenes", which are stored alongside each
        other at different coordinates in the image plane, often separated by empty
        space. Specifying this argument will read only the single scene with that index
        from the file. Think of it as cropping the CZI file to the region where that
        scene is located.

        If no scene index is specified, the entire image will be read. In case it
        contains multiple scenes, they will all be present in the resulting image.
        This is usually not desirable due to the empty space between them.
        In general, only omit this argument or set it to `None` if you know that
        your CZI file does not contain any scenes.

        The static function :py:meth:`get_bounding_rectangles` can be used to find out
        how many scenes a given file contains and what their bounding rectangles are.

        The scene can also be provided as part of `data_path` by appending an `"@"`
        followed by the scene index to the filename.

    depth_axis : {"none", "Z", "T"}, default: "none"
        Which axis to use as depth-axis for providing 3-D patches.

        - `"none"`: Only provide 2-D patches. If a Z or T dimension is present in the
          data, they will be combined into the sample dimension `S`.
        - `"Z"`: Use the Z-axis as depth-axis. If a T axis is present as well, it will
          be merged into the sample dimensions `S`.
        - `"T"`: Use the T-axis as depth-axis. If a Z axis is present as well, it will
          be merged into the sample dimensions `S`.

    Attributes
    ----------
    source : Path
        Path to the CZI file, including the scene index if specified.
    data_path : Path
        Path to the CZI file without scene index.
    scene : int or None
        Index of the scene to extract, or None if not specified.
    data_shape : Sequence[int]
        The shape of the data in the order `(SC(Z)YX)`.
    axes : str
        The axes in the CZI file corresponding to the dimensions in `data_shape`.
        The following values can occur:

        - "SCZYX" for 3-D volumes if `depth_axis` is `"Z"`.
        - "SCTYX" for time-series if `depth_axis` is `"T"`.
        - "SCYX" if `depth_axis` is `"none"`.

        The axis `S` (sample) is the only one not mapping one-to-one to an axis in the
        CZI file but combines all remaining axes present in the file into one.

    Examples
    --------
    Create an image stack for the first scene in a CZI file:
    >>> stack = CziImageStack("path/to/file.czi", scene=0)  # doctest: +SKIP

    Alternatively, the scene index can also be provided as part of the filename.
    This is mainly intended for re-creating an image stack from the `source` property:
    >>> stack = CziImageStack("path/to/file.czi@0")  # doctest: +SKIP
    >>> stack2 = CziImageStack(stack.source)  # doctest: +SKIP

    If the CZI file contains a third dimension (Z or T) and you want to perform 3-D
    denoising, you need to explicitly set `depth_axis` to `"Z"` or `"T"`:
    >>> stack_2d = CziImageStack("path/to/file.czi", scene=0)  # doctest: +SKIP
    >>> stack_2d.axes, stack_2d.data_shape  # doctest: +SKIP
    ('SCYX', [40, 1, 512, 512])
    >>> stack_3d = CziImageStack(  # doctest: +SKIP
    ...     "path/to/file.czi", scene=0, depth_axis="Z"
    ... )
    >>> stack_3d.axes, stack_3d.data_shape  # doctest: +SKIP
    ('SCZYX', [4, 1, 10, 512, 512])
    """

    def __init__(
        self,
        data_path: str | Path,
        scene: int | None = None,
        depth_axis: Literal["none", "Z", "T"] = "none",
    ) -> None:
        if not pyczi_available:
            raise ImportError(
                "The CZI image stack requires the `pylibCZIrw` package to be installed."
                " Please install it with `pip install careamics[czi]`."
            )

        _data_path = Path(data_path)

        # Check for scene encoded in filename.
        # Normally, file path and scene should be provided as separate arguments but
        # we would also like to support using the `source` property to re-create the
        # CZI image stack. In this case, the scene index is encoded in the file path.
        scene_matches = re.match(r"^(.*)@(\d+)$", _data_path.name)
        if scene_matches:
            if scene is not None:
                raise ValueError(
                    f"Scene index is specified in the filename ({_data_path.name}) and "
                    f"as an argument ({scene}). Please specify only one."
                )
            _data_path = _data_path.parent / scene_matches.group(1)
            scene = int(scene_matches.group(2))

        # Set variables
        self.data_path = _data_path
        self.scene = scene
        self._depth_axis = depth_axis

        # Open CZI file
        self._czi = CziReader(str(self.data_path))

        # Determine metadata
        self.axes, self.data_shape, self._bounding_rectangle, self._sample_axes = (
            self._get_shape()
        )
        self.data_dtype = np.float32

    def __del__(self):
        if hasattr(self, "_czi"):
            # Close CZI file
            self._czi.close()

    def __getstate__(self) -> dict[str, Any]:
        # Remove CziReader object from state to avoid pickling issues
        state = self.__dict__.copy()
        del state["_czi"]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        # Reopen CZI file after unpickling
        self.__dict__.update(state)
        self._czi = CziReader(str(self.data_path))

    # TODO: we append the scene index to the file name
    #       - not sure if this is a good approach
    @property
    def source(self) -> Path:
        filename = self.data_path.name
        if self.scene is not None:
            filename = f"{filename}@{self.scene}"
        return self.data_path.parent / filename

    def extract_patch(
        self, sample_idx: int, coords: Sequence[int], patch_size: Sequence[int]
    ) -> NDArray:
        # Determine 3rd dimension (T, Z or none)
        if len(coords) == 3:
            if len(self.axes) != 5:
                raise ValueError(
                    f"Requested a 3D patch from a 2D image stack with axes {self.axes}."
                )
            third_dim = self.axes[2]
            third_dim_offset, third_dim_size = coords[0], patch_size[0]
        else:
            if len(self.axes) != 4:
                raise ValueError(
                    f"Requested a 2D patch from a 3D image stack with axes {self.axes}."
                )
            third_dim = None
            third_dim_offset, third_dim_size = 0, 1

        # Set up ROI to extract from each plane as (x, y, w, h)
        roi = (
            self._bounding_rectangle.x + coords[-1],
            self._bounding_rectangle.y + coords[-2],
            patch_size[-1],
            patch_size[-2],
        )

        # Create output array of shape (C, Z, Y, X)
        patch = np.empty(
            (self.data_shape[1], third_dim_size, *patch_size[-2:]), dtype=np.float32
        )

        # Set up plane to index `sample_idx`
        sample_shape = list(self._sample_axes.values())
        sample_indices = np.unravel_index(sample_idx, sample_shape)
        plane = {
            dimension: int(index)
            for dimension, index in zip(
                self._sample_axes.keys(), sample_indices, strict=False
            )
        }

        # Read XY planes sequentially
        for channel in range(self.data_shape[1]):
            for third_dim_index in range(third_dim_size):
                plane["C"] = channel
                if third_dim is not None:
                    plane[third_dim] = third_dim_offset + third_dim_index
                extracted_roi = self._czi.read(roi=roi, plane=plane, scene=self.scene)
                if extracted_roi.ndim == 3:
                    if extracted_roi.shape[-1] > 1:
                        raise ValueError(
                            "CZI files with RGB channels are currently not supported."
                        )
                    extracted_roi = extracted_roi.squeeze(-1)
                patch[channel, third_dim_index] = extracted_roi

        # Remove dummy 3rd dimension for 2-D data
        if third_dim is None:
            patch = patch.squeeze(1)

        return patch

    def _get_shape(self) -> tuple[str, list[int], Rectangle, dict[str, int]]:
        """Determines the shape of the selected scene.

        Returns
        -------
        axes : str
            String specifying the axis order. Examples:

            - "SCZYX" for 3-D volumes if `depth_axis` is `"Z"`.
            - "SCTYX" for time-series if `depth_axis` is `"T"`.
            - "SCYX" if `depth_axis` is `"none"`.

            The axis `S` is the sample dimension and combines all remaining axes
            present in the data.

        shape : list[int]
            The size of each axis, in the order listed in `axes`.

        bounding_rectangle : Rectangle
            The bounding rectangle of the scene in pixels. The rectangle is
            defined by its top-left corner (x, y) and its width and height (w, h).

        sample_axes : dict[str, int]
            A dictionary with information about the remaining axes used for the
            sample dimension.
            The keys are the axis names (e.g., "T", "Z") and the values are their
            respective sizes.
        """
        # Get CZI dimensions
        total_bbox = self._czi.total_bounding_box_no_pyramid
        if self.scene is None:
            bounding_rectangle = self._czi.total_bounding_rectangle_no_pyramid
        else:
            bounding_rectangle = self._czi.scenes_bounding_rectangle_no_pyramid[
                self.scene
            ]

        # Determine if T and Z axis are present
        # Note: An axis of size 1 is as good as no axis since we cannot use it for 3-D
        # denoising.
        has_time = "T" in total_bbox and (total_bbox["T"][1] - total_bbox["T"][0]) > 1
        has_depth = "Z" in total_bbox and (total_bbox["Z"][1] - total_bbox["Z"][0]) > 1

        # Determine axis order depending on `depth_axis`
        if self._depth_axis == "Z":
            axes = "SCZYX"
            if not has_depth:
                raise RuntimeError(
                    f"The CZI file {self.data_path} does not contain a Z axis to use "
                    'for 3-D denoising. Consider setting `axes="YX"` or '
                    '`depth_axis="none"` to perform 2-D denoising instead.'
                )
        elif self._depth_axis == "T":
            axes = "SCTYX"
            if not has_time:
                raise RuntimeError(
                    f"The CZI file {self.data_path} does not contain a T axis to use "
                    'for 3-D denoising. Consider setting `axes="YX"` or '
                    '`depth_axis="none"` to perform 2-D denoising instead.'
                )
        else:
            axes = "SCYX"

        # Calculcate size of sample dimension S, combining all axes not used elsewhere.
        # This could, for example, be a time axis. If we only perform 2-D denoising, a
        # potentially present Z axis would also be used as sample dimension. If both,
        # T and Z, are present, both need to be combined into the sample dimension.
        # The same needs to be done to any other potentially present axis in the CZI
        # file which is not a spatial or channel axis.
        # The following code calculates the size of the combined sample axis.
        sample_axes = {}
        sample_size = 1
        for dimension, (start, end) in total_bbox.items():
            if dimension not in axes:
                sample_axes[dimension] = end - start
                sample_size *= end - start

        # Determine data shape
        shape = []
        for dimension in axes:
            if dimension == "S":
                shape.append(sample_size)
            elif dimension == "Y":
                shape.append(bounding_rectangle.h)
            elif dimension == "X":
                shape.append(bounding_rectangle.w)
            elif dimension in total_bbox:
                shape.append(total_bbox[dimension][1] - total_bbox[dimension][0])
            else:
                shape.append(1)

        return axes, shape, bounding_rectangle, sample_axes

    @classmethod
    def get_bounding_rectangles(
        cls, czi: Path | str | CziReader
    ) -> dict[int | None, Rectangle]:
        """Gets the bounding rectangles of all scenes in a CZI file.

        Parameters
        ----------
        czi : Path or str or pyczi.CziReader
            Path to the CZI file or an already opened file as CziReader object.

        Returns
        -------
        dict[int | None, Rectangle]
            A dictionary mapping scene indices to their bounding rectangles in the
            format `(x, y, w, h)`.
            If no scenes are present in the CZI file, the returned dictionary will
            have only one entry with key `None`, whose bounding rectangle covers the
            entire image.
        """
        if not isinstance(czi, CziReader):
            with open_czi(str(czi)) as czi_reader:
                return cls.get_bounding_rectangles(czi_reader)

        scenes_bounding_rectangle = czi.scenes_bounding_rectangle_no_pyramid
        if len(scenes_bounding_rectangle) >= 1:
            # Ensure keys are int | None for type compatibility
            return {int(k): v for k, v in scenes_bounding_rectangle.items()}
        else:
            return {None: czi.total_bounding_rectangle_no_pyramid}
