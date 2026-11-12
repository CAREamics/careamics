#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Any, Literal

from careamics import CAREamist
from careamics.config.support import SupportedData


def without_none(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in kwargs.items() if value is not None}


def prediction_careamist(
    ckpt_path: Path,
    data_path: str,
    batch_size: int,
    tile_size: tuple[int, ...],
    tile_overlap: tuple[int, ...],
    axes: str,
    data_type: SupportedData,
    write_type: Literal["tiff", "zarr", "custom"],
    output_path: Path,
):
    """Function to denoised a dataset according to a pretrained model using careamics"""
    careamics_pretrained = CAREamist(checkpoint_path=ckpt_path)
    careamics_pretrained.predict_to_disk(
        pred_data=data_path,
        prediction_dir=output_path / "predictions",
        **without_none(
            {
                "batch_size": batch_size,
                "tile_size": tile_size,
                "tile_overlap": tile_overlap,
                "axes": axes,
                "data_type": data_type if data_type is None else data_type.value,
                "write_type": write_type,
            }
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path", type=Path, required=True, help="Path to pretrained model"
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Path to folder with images"
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        help="Path to save the output files.",
        default=Path("."),
    )
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument(
        "--tile_size", nargs="+", type=int, default=None, help="2D or 3D"
    )
    parser.add_argument(
        "--tile_overlap", nargs="+", type=int, default=None, help="2D or 3D"
    )
    parser.add_argument("--axes", type=str, default=None)
    parser.add_argument("--data_type", type=SupportedData, default=None)
    parser.add_argument(
        "--write_type", choices=["tiff", "zarr", "custom"], default=None
    )
    args = parser.parse_args()

    prediction_careamist(
        args.ckpt_path,
        args.data,
        args.batch_size,
        args.tile_size,
        args.tile_overlap,
        args.axes,
        args.data_type,
        args.write_type,
        args.output_path,
    )
