#!/usr/bin/env python
"""Export middle-time, middle-Z slices from every OME-Zarr pyramid level."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import zarr
from PIL import Image

DEFAULT_INPUT = Path(__file__).parent / "data/001-small-lowT/raw.ome.zarr"
DEFAULT_OUTPUT = Path(__file__).parent / "data/001-small-lowT/middle_slices"


def export_middle_slices(
    input_path: Path,
    output_dir: Path,
    *,
    display_min: float = 0.0,
    display_max: float = 1500.0,
) -> None:
    """Export channel 0 at the middle time and Z position for each level."""
    if display_max <= display_min:
        raise ValueError("display_max must be greater than display_min")

    output_dir.mkdir(parents=True, exist_ok=True)
    for level in range(5):
        array = zarr.open_array(input_path / str(level), mode="r")
        if array.ndim != 5:
            raise ValueError(f"Expected a TCZYX array, got shape {array.shape}")

        time_index = array.shape[0] // 2
        z_index = array.shape[2] // 2
        image = array[time_index, 0, z_index, :, :]
        image = np.clip(image, display_min, display_max)
        image = ((image - display_min) * 255 / (display_max - display_min)).astype(
            np.uint8
        )

        output_path = output_dir / (f"level_{level}_t{time_index}_c0_z{z_index}.png")
        Image.fromarray(image).save(output_path)
        print(f"Wrote {output_path} ({image.shape[1]} x {image.shape[0]})")


def main() -> None:
    """Parse command-line arguments and export the slices."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", dest="input_path", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output-dir", dest="output_dir", type=Path, default=DEFAULT_OUTPUT
    )
    parser.add_argument("--display-min", type=float, default=0.0)
    parser.add_argument("--display-max", type=float, default=1500.0)
    args = parser.parse_args()
    export_middle_slices(**vars(args))


if __name__ == "__main__":
    main()
