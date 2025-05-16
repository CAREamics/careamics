import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pylibCZIrw.czi import CziReader, Rectangle, open_czi


class CziImageStack:
    """
    A class for extracting patches from an image stack that is stored as a CZI file.

    Parameters
    ----------
    data_path : str or Path
        Path to the CZI file.

    scene : int, optional
        Index of the scene to extract. If not specified, the entire image covering
        all scenes present in the CZI file will be extracted.
        The scene can also be provided as part of `data_path` by appending an `"@"`
        followed by the scene index to the filename.

    depth_axis : {"none", "Z", "T", "auto"}, default: "auto"
        Which axis to use as depth-axis for providing 3-D patches.

        - `"none"`: Only provide 2-D patches. If a Z or T dimension is present in the
          data, they will be combined into the sample dimension `S`.
        - `"Z"`: Use the Z-axis as depth-axis. If a T axis is present as well, it will
          be used as sample dimensions `S`.
        - `"T"`: Use the T-axis as depth-axis. If a Z axis is present as well, it will
          be used as sample dimensions `S`.
        - `"auto"`: Automatically uses a Z or T axis present in the data as depth axis.
          If both are present, the Z axis takes precedence.

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
        Possible axes are `C`, `T`, `Z`, `Y`, `X`, and `S`. The axis `S` (sample)
        is special and combines all remaining axes present in the data.
    """

    def __init__(
        self,
        data_path: str | Path,
        scene: int | None = None,
        depth_axis: Literal["none", "Z", "T", "auto"] = "auto",
    ) -> None:
        _data_path = Path(data_path)

        # Check for scene encoded in filename.
        # Normally, file path and scene should be provided as separate arguments but
        # we would also like to support using the `source` property to re-create the
        # CZI image stack. In this case, the scene index is encoded in the file path.
        scene_matches = re.match(r"^(.*)@(\d+)$", _data_path.name)
        if scene_matches:
            if scene is not None:
                raise ValueError(
                    "Scene index is specified in the filename and as an argument. "
                    "Please specify only one."
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
            for dimension, index in zip(self._sample_axes.keys(), sample_indices)
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

            - "SCZYX" for 3-D data if `depth_axis` is `"Z"`.
            - "SCTYX" for 3-D data if `depth_axis` is `"T"`.
            - "SCYX" for 2-D time-series if `depth_axis` is `"none"`.
            - "SCTYX" for 2-D time-series if `depth_axis` is `"T"`.
            - "SCYX" for 2-D images.

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

        # Determine whether to use time as depth dimension
        depth_axis = self._depth_axis
        if depth_axis == "auto":
            if has_depth:
                depth_axis = "Z"
            elif has_time:
                depth_axis = "T"
            else:
                depth_axis = "none"

        # Determine axis order depending on data type and `depth_axis`
        if depth_axis == "Z" and has_depth:
            axes = "SCZYX"
        elif depth_axis == "T" and has_time:
            axes = "SCTYX"
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
