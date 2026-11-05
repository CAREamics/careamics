#!/usr/bin/env python
"""Export a small GIF timelapse from the first channel of an OME-Zarr image.

The example dataset has axes ``TCZYX``. This script exports the time axis after
selecting channel 0 and the middle Z-slice.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import zarr
from PIL import Image

DEFAULT_INPUT = Path(__file__).parent / "data/001-small-lowT/raw.ome.zarr"
DEFAULT_OUTPUT = Path(__file__).parent / "data/001-small-lowT/timelapse.gif"


def export_timelapse(
    input_path: Path,
    output_path: Path,
    *,
    level: int = 0,
    frame_step: int = 1,
    fps: float = 8.0,
    max_size: int = 512,
    display_min: float = 0.0,
    display_max: float = 1500.0,
) -> None:
    """Export channel 0 and the middle Z-slice as an animated GIF.

    Parameters
    ----------
    input_path : Path
        Path to the OME-Zarr store.
    output_path : Path
        Destination GIF path.
    level : int, default=0
        Resolution level in the OME-Zarr multiscale pyramid.
    frame_step : int, default=1
        Keep every ``frame_step``-th time point.
    fps : float, default=8.0
        Playback rate of the GIF.
    max_size : int, default=512
        Maximum width or height of each output frame in pixels. Set to 0 to
        keep the source resolution.
    display_min, display_max : float
        Intensity range mapped to black and white.
    """
    if frame_step < 1:
        raise ValueError("frame_step must be at least 1")
    if fps <= 0:
        raise ValueError("fps must be greater than 0")
    if display_max <= display_min:
        raise ValueError("display_max must be greater than display_min")

    array = zarr.open_array(input_path / str(level), mode="r")
    if array.ndim != 5:
        raise ValueError(f"Expected a TCZYX array, got shape {array.shape}")

    middle_z = array.shape[2] // 2
    frames = array[::frame_step, 0, middle_z, :, :]
    frames = np.clip(frames, display_min, display_max)
    frames = ((frames - display_min) * 255 / (display_max - display_min)).astype(
        np.uint8
    )

    pil_frames: list[Image.Image] = []
    for frame in frames:
        image = Image.fromarray(frame, mode="L")
        if max_size > 0 and max(image.size) > max_size:
            scale = max_size / max(image.size)
            size = tuple(round(value * scale) for value in image.size)
            image = image.resize(size, Image.Resampling.LANCZOS)
        pil_frames.append(image)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=round(1000 / fps),
        loop=0,
        optimize=True,
    )
    print(
        f"Wrote {len(pil_frames)} frames to {output_path} "
        f"(level {level}, channel 0, z={middle_z})"
    )


def main() -> None:
    """Parse command-line arguments and export the timelapse."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", dest="input_path", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output", dest="output_path", type=Path, default=DEFAULT_OUTPUT
    )
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--max-size", type=int, default=512)
    parser.add_argument("--display-min", type=float, default=0.0)
    parser.add_argument("--display-max", type=float, default=1500.0)
    args = parser.parse_args()
    export_timelapse(**vars(args))


if __name__ == "__main__":
    main()
