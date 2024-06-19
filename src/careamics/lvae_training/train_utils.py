"""
This script contains the utility functions for training the LVAE model.
These functions are mainly used in `train.py` script.
"""

import logging
import os
import pickle
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import ml_collections


def log_config(config: ml_collections.ConfigDict, cur_workdir: str) -> None:
    # Saving config file.
    with open(os.path.join(cur_workdir, "config.pkl"), "wb") as f:
        pickle.dump(config, f)
    print(f"Saved config to {cur_workdir}/config.pkl")


def set_logger(workdir: str) -> None:
    os.makedirs(workdir, exist_ok=True)
    fstream = open(os.path.join(workdir, "stdout.txt"), "w")
    handler = logging.StreamHandler(fstream)
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel("INFO")


def get_new_model_version(model_dir: str) -> str:
    """
    A model will have multiple runs. Each run will have a different version.
    """
    versions = []
    for version_dir in os.listdir(model_dir):
        try:
            versions.append(int(version_dir))
        except:
            print(
                f"Invalid subdirectory:{model_dir}/{version_dir}. Only integer versions are allowed"
            )
            exit()
    if len(versions) == 0:
        return "0"
    return f"{max(versions) + 1}"


def get_model_name(config: ml_collections.ConfigDict) -> str:
    return "LVAE_denoiSplit"


def get_workdir(
    config: ml_collections.ConfigDict,
    root_dir: str,
    use_max_version: bool,
    nested_call: int = 0,
):
    rel_path = datetime.now().strftime("%y%m")
    cur_workdir = os.path.join(root_dir, rel_path)
    Path(cur_workdir).mkdir(exist_ok=True)

    rel_path = os.path.join(rel_path, get_model_name(config))
    cur_workdir = os.path.join(root_dir, rel_path)
    Path(cur_workdir).mkdir(exist_ok=True)

    if use_max_version:
        # Used for debugging.
        version = int(get_new_model_version(cur_workdir))
        if version > 0:
            version = f"{version - 1}"

        rel_path = os.path.join(rel_path, str(version))
    else:
        rel_path = os.path.join(rel_path, get_new_model_version(cur_workdir))

    cur_workdir = os.path.join(root_dir, rel_path)
    try:
        Path(cur_workdir).mkdir(exist_ok=False)
    except FileExistsError:
        print(
            f"Workdir {cur_workdir} already exists. Probably because someother program also created the exact same directory. Trying to get a new version."
        )
        time.sleep(2.5)
        if nested_call > 10:
            raise ValueError(
                f"Cannot create a new directory. {cur_workdir} already exists."
            )

        return get_workdir(config, root_dir, use_max_version, nested_call + 1)

    return cur_workdir, rel_path


def get_mean_std_dict_for_model(config, train_dset):
    """
    Computes the mean and std for the model. This will be subsequently passed to the model.
    """
    mean_dict, std_dict = train_dset.get_mean_std()

    return deepcopy(mean_dict), deepcopy(std_dict)


class MetricMonitor:
    def __init__(self, metric):
        assert metric in ["val_loss", "val_psnr"]
        self.metric = metric

    def mode(self):
        if self.metric == "val_loss":
            return "min"
        elif self.metric == "val_psnr":
            return "max"
        else:
            raise ValueError(f"Invalid metric:{self.metric}")
