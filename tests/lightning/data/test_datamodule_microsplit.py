"""Tests for MicroSplit dataset dispatch in `CareamicsDataModule`.

`CareamicsDataModule.setup` selects how datasets are built from the type of the data
configuration: a `MicroSplitDataConfig` builds MicroSplit datasets, with lateral
context, while any other `DataConfig` builds the basic patch datasets used by CARE,
N2N, N2V and HDN.

Note that `test_microsplit_train_datamodule.py` covers the unrelated legacy
`careamics.lightning.data.microsplit_data_module` and its own, different,
`MicroSplitDataConfig`.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import tifffile

from careamics.config.data import DataConfig, MicroSplitDataConfig
from careamics.dataset.patch_constructor import BasicPatchConstr
from careamics.dataset.patch_constructor.microsplit_patch_constructors import (
    PairedInputTargetMsPatchConstr,
    PredMsPatchConstr,
)
from careamics.lightning.data.data_module import CareamicsDataModule
from careamics.lightning.data.grouped_index_sampler import GroupedIndexSampler

PATCH_SIZE = (16, 16)


def _config(microsplit: bool = True, **kwargs: Any) -> DataConfig:
    """Return a training data configuration, MicroSplit or generic."""
    params: dict[str, Any] = {
        "mode": "training",
        "data_type": "array",
        "axes": "SCYX",
        "patching": {"name": "stratified", "patch_size": PATCH_SIZE, "seed": 42},
        "normalization": {"name": "mean_std"},
        "batch_size": 2,
        "seed": 42,
    }
    if microsplit:
        params["multiscale_count"] = 1
    params.update(kwargs)
    return (MicroSplitDataConfig if microsplit else DataConfig)(**params)


def _paired_arrays(
    value: float = 1.0, n_channels: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Return a single-channel input and its multi-channel target."""
    target = np.full((2, n_channels, 32, 32), value, dtype=np.float32)
    return target.sum(axis=1, keepdims=True), target


def _datamodule(config: DataConfig, **kwargs: Any) -> CareamicsDataModule:
    """Return a data module built from paired training and validation arrays."""
    train_input, train_target = _paired_arrays()
    val_input, val_target = _paired_arrays()
    data: dict[str, Any] = {
        "train_data": train_input,
        "train_data_target": train_target,
        "val_data": val_input,
        "val_data_target": val_target,
    }
    data.update(kwargs)
    return CareamicsDataModule(config, **data)


@pytest.mark.parametrize("multiscale_count", [1, 3])
def test_fit_builds_microsplit_datasets(multiscale_count: int) -> None:
    """Test a MicroSplit configuration builds datasets with lateral context."""
    # Arrange
    datamodule = _datamodule(_config(multiscale_count=multiscale_count))

    # Act
    datamodule.setup("fit")

    # Assert
    assert isinstance(
        datamodule.train_dataset.patch_constructor, PairedInputTargetMsPatchConstr
    )
    assert isinstance(
        datamodule.val_dataset.patch_constructor, PairedInputTargetMsPatchConstr
    )

    input_region, target_region = datamodule.train_dataset[0]
    assert input_region.data.shape == (multiscale_count, *PATCH_SIZE)
    assert target_region.data.shape == (2, *PATCH_SIZE)


def test_fit_builds_basic_datasets_for_generic_config() -> None:
    """Test a plain configuration still builds basic patch datasets."""
    datamodule = _datamodule(_config(microsplit=False))

    datamodule.setup("fit")

    assert isinstance(datamodule.train_dataset.patch_constructor, BasicPatchConstr)
    assert isinstance(datamodule.val_dataset.patch_constructor, BasicPatchConstr)


def test_fit_computes_statistics_on_training_data() -> None:
    """Test validation reuses the training statistics rather than its own.

    The training dataset must be built before the validation configuration is derived.
    """
    # Arrange
    train_input, train_target = _paired_arrays(value=1.0)
    val_input, val_target = _paired_arrays(value=100.0)
    config = _config()
    datamodule = CareamicsDataModule(
        config,
        train_data=train_input,
        train_data_target=train_target,
        val_data=val_input,
        val_data_target=val_target,
    )

    # Act
    datamodule.setup("fit")

    # Assert: the input sums the two training channels, so its mean is 2.0, whereas
    # the validation data would give 200.0.
    assert config.normalization.input_means == [2.0]
    assert (
        datamodule.val_dataset.config.normalization.input_means
        == config.normalization.input_means
    )


def test_fit_saves_resolved_statistics_to_hparams() -> None:
    """Test the statistics computed during setup are saved to the hyperparameters."""
    datamodule = _datamodule(_config())

    datamodule.setup("fit")

    normalization = datamodule.hparams["data_config"]["normalization"]
    assert normalization["input_means"] == [2.0]
    assert len(normalization["target_means"]) == 2


def test_fit_rejects_validation_splitting() -> None:
    """Test automatic validation splitting is not supported for MicroSplit."""
    train_input, train_target = _paired_arrays()
    datamodule = CareamicsDataModule(
        _config(), train_data=train_input, train_data_target=train_target
    )

    with pytest.raises(NotImplementedError, match="Automatic validation splitting"):
        datamodule.setup("fit")


@pytest.mark.parametrize(
    ("missing_field", "match"),
    [
        ("train_data_target", "MicroSplit is supervised"),
        ("val_data_target", "`val_data_target` must be provided"),
    ],
)
def test_fit_requires_targets(missing_field: str, match: str) -> None:
    """Test MicroSplit requires both training and validation targets."""
    datamodule = _datamodule(_config(), **{missing_field: None})

    with pytest.raises(ValueError, match=match):
        datamodule.setup("fit")


def test_predict_builds_microsplit_dataset() -> None:
    """Test a MicroSplit configuration builds a MicroSplit prediction dataset."""
    pred_input, _ = _paired_arrays()
    datamodule = CareamicsDataModule(
        _config(mode="predicting", patching={"name": "whole"}), pred_data=pred_input
    )

    datamodule.setup("predict")

    assert isinstance(datamodule.predict_dataset.patch_constructor, PredMsPatchConstr)


def test_predict_converts_training_config() -> None:
    """Test a training configuration is converted to whole-image prediction."""
    pred_input, _ = _paired_arrays()
    datamodule = CareamicsDataModule(_config(), pred_data=pred_input)

    datamodule.setup("predict")

    assert datamodule.predict_dataset.config.mode == "predicting"
    assert isinstance(datamodule.predict_dataset.patch_constructor, PredMsPatchConstr)


def test_predict_builds_basic_dataset_for_generic_config() -> None:
    """Test a plain configuration still builds a basic prediction dataset."""
    pred_input, _ = _paired_arrays(n_channels=1)
    datamodule = CareamicsDataModule(_config(microsplit=False), pred_data=pred_input)

    datamodule.setup("predict")

    assert isinstance(datamodule.predict_dataset.patch_constructor, BasicPatchConstr)


def test_dictionary_with_microsplit_fields_is_rejected() -> None:
    """Test a MicroSplit dictionary is rejected rather than silently downgraded.

    Dictionaries are validated as a plain `DataConfig`, which would drop the MicroSplit
    fields and build a dataset without lateral context.
    """
    config_dict = _config().model_dump()
    train_input, train_target = _paired_arrays()

    with pytest.raises(ValueError, match="MicroSplit-only field"):
        CareamicsDataModule(
            config_dict, train_data=train_input, train_data_target=train_target
        )


def test_generic_dictionary_is_accepted() -> None:
    """Test a plain configuration dictionary is still accepted."""
    config_dict = _config(microsplit=False).model_dump()
    train_input, train_target = _paired_arrays()

    datamodule = CareamicsDataModule(
        config_dict, train_data=train_input, train_data_target=train_target
    )

    assert isinstance(datamodule.config, DataConfig)


def test_sampler_for_non_in_memory_tiff(tmp_path: Path) -> None:
    """Test grouped sampling is used for MicroSplit datasets read from files."""
    # Arrange
    # source and target files must share their names, hence the two folders
    input_dir = tmp_path / "inputs"
    target_dir = tmp_path / "targets"
    input_dir.mkdir()
    target_dir.mkdir()

    input_paths, target_paths = [], []
    for index in range(2):
        _, target = _paired_arrays()
        input_path = input_dir / f"image_{index}.tiff"
        target_path = target_dir / f"image_{index}.tiff"
        tifffile.imwrite(
            input_path, target.sum(axis=1, keepdims=True), metadata={"axes": "SCYX"}
        )
        tifffile.imwrite(target_path, target, metadata={"axes": "SCYX"})
        input_paths.append(input_path)
        target_paths.append(target_path)

    datamodule = CareamicsDataModule(
        _config(data_type="tiff", in_memory=False),
        train_data=input_paths,
        train_data_target=target_paths,
        val_data=input_paths,
        val_data_target=target_paths,
    )

    # Act
    datamodule.setup("fit")

    # Assert
    assert isinstance(datamodule._sampler("train"), GroupedIndexSampler)
    assert next(iter(datamodule.train_dataloader())) is not None
