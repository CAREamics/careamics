#!/usr/bin/env python3

"""Script for the CLI. Runs training."""

import argparse
import json
from pathlib import Path
from typing import Any

from lightning.pytorch.callbacks import ModelCheckpoint

from careamics import CAREamist
from careamics.config.utils.configuration_io import load_configuration


def without_none(kwargs):
    return {key: value for key, value in kwargs.items() if value is not None}


def get_checkpoint_record_data(careamist: CAREamist) -> dict[str, Any]:
    """Get a record of the checkpoints produced during training.

    Parameters
    ----------
    careamist : CAREamist
        A CAREamist instance that has been used for training.

    Returns
    -------
    dict[str, Any]
        Contains a record of the checkpoints produced during training.

    Raises
    ------
    ValueError
        If no checkpoints were found.
    ValueError
        If the monitoring mode in `ModelCheckpoint` is not `"min"` or `"max"`.
    """
    model_ckpt_callback = careamist.trainer.checkpoint_callback
    assert isinstance(model_ckpt_callback, ModelCheckpoint)

    data: dict[str, Any] = {}

    last = model_ckpt_callback.last_model_path
    data["last"] = Path(last).name if last else None

    best = model_ckpt_callback.best_model_path
    data["best"] = Path(best).name if best else None

    data["monitor"] = model_ckpt_callback.monitor
    data["mode"] = model_ckpt_callback.mode

    # add best k, put in a list with best at the first element
    best_k = [
        {"name": Path(ckpt_path).name, "score": score.item()}
        for ckpt_path, score in model_ckpt_callback.best_k_models.items()
    ]
    if data["mode"] == "min":
        best_k = sorted(best_k, key=lambda x: x["score"])
    elif data["mode"] == "max":
        best_k = sorted(best_k, key=lambda x: x["score"], reverse=True)
    else:
        raise ValueError(f"Unrecognized checkpoint monitoring mode: {data['mode']}")
    data["best_k"] = best_k

    # all checkpoints
    data["checkpoints"] = [ckpt_path.name for ckpt_path in careamist.get_checkpoints()]
    if len(data["checkpoints"]) == 0:
        raise ValueError(
            "No checkpoints found. Ensure that model checkpointing has been configured "
            "correctly."
        )

    # select checkpoint that should be used
    select = careamist._get_default_ckpt(None)  # will only return "best" or "last"
    if select == "last":
        selected = data["checkpoints"][-1]  # last saved
    else:
        selected = data[str(select)]
    data["selected"] = selected

    return data


def save_checkpoint_record(careamist: CAREamist) -> None:
    """Save a `checkpoint_record.json` file.

    Parameters
    ----------
    careamist : CAREamist
        A CAREamist instance that has been used for training.
    """
    model_ckpt_callback = careamist.trainer.checkpoint_callback
    assert isinstance(model_ckpt_callback, ModelCheckpoint)
    assert model_ckpt_callback.dirpath is not None  # CAREamist always sets this
    dirpath = Path(model_ckpt_callback.dirpath)

    data = get_checkpoint_record_data(careamist)
    with open(dirpath / "checkpoint_record.json", "w") as f:
        json.dump(data, f, indent=2)


def train():
    """Function for the CLI to run training."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_path", type=Path, required=True, help="Path to save the output files."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--train_data", type=str, required=True, help="Path to train data."
    )
    parser.add_argument(
        "--train_target", type=str, default=None, help="Path to target data."
    )
    parser.add_argument(
        "--val_data", type=str, default=None, help="Path to validation data."
    )
    parser.add_argument(
        "--val_target", type=str, default=None, help="Path to validation target data."
    )
    # TODO: add filtering mask?

    args = parser.parse_args()

    config = load_configuration(args.config)
    # TODO: have to make sure that checkpoint callback is saving last checkpoint?
    # TODO: export last/best checkpoint name?

    careamist = CAREamist(config=config, work_dir=args.output_path)
    careamist.train(
        train_data=args.train_data,
        train_data_target=args.train_target,
        **without_none({"val_data": args.val_data, "val_data_target": args.val_target}),
    )
    save_checkpoint_record(careamist)


if __name__ == "__main__":
    train()
