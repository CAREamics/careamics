"""MicroSplit Lightning DataModule for the NG dataset pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytorch_lightning as L
from numpy.typing import NDArray
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from careamics.config.data.ng_microsplit_config import MicroSplitDataConfig
from careamics.dataset_ng.microsplit_dataset import MicroSplitDataset
from careamics.lightning.dataset_ng.data_module import CareamicsDataModule


class MicroSplitDataModule(CareamicsDataModule):
    """Lightning DataModule for MicroSplit (LVAE) training.

    Extends :class:`CareamicsDataModule` to create :class:`MicroSplitDataset`
    instances instead of :class:`CareamicsDataset` when the configuration is a
    :class:`MicroSplitDataConfig`.

    All sampler / collate behaviour is inherited unchanged from the parent.

    Parameters
    ----------
    data_config : MicroSplitDataConfig
        MicroSplit-specific dataset configuration.
    train_data : numpy.ndarray
        Training array in ``(S, C, Y, X)`` order.
    val_data : numpy.ndarray or None, optional
        Validation array in ``(S, C, Y, X)`` order.  If ``None``, a random
        split of ``n_val_patches`` patches from the training data is used
        (requires stratified patching in config).
    """

    def __init__(
        self,
        data_config: MicroSplitDataConfig,
        train_data: NDArray,
        val_data: NDArray | None = None,
        **kwargs: Any,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        data_config : MicroSplitDataConfig
            Data configuration for MicroSplit training.
        train_data : numpy.ndarray
            Training array in ``(S, C, Y, X)`` order.
        val_data : numpy.ndarray or None, optional
            Validation array. If ``None``, validation patches are split from
            training data using ``n_val_patches`` (default = 8).
        **kwargs : Any
            Additional keyword arguments forwarded to
            :class:`CareamicsDataModule` (e.g. ``loading``).
        """
        if not isinstance(data_config, MicroSplitDataConfig):
            raise TypeError(
                f"MicroSplitDataModule requires a MicroSplitDataConfig, "
                f"got {type(data_config).__name__}."
            )

        # Store raw arrays; don't call super().__init__() here because the
        # parent class validates data against NGDataConfig field expectations
        # and would wire up CareamicsDataset rather than MicroSplitDataset.
        # We initialise only what we need from L.LightningDataModule.
        L.LightningDataModule.__init__(self)

        self.config = data_config
        self.batch_size: int = data_config.batch_size

        self._train_data = train_data
        self._val_data = val_data

        self.train_dataset: MicroSplitDataset | None = None
        self.val_dataset: MicroSplitDataset | None = None
        self.predict_dataset = None

        self.rng = np.random.default_rng(seed=data_config.seed)

    def setup(self, stage: str) -> None:
        """Create datasets for training/validation.

        Parameters
        ----------
        stage : str
            One of ``"fit"``, ``"validate"``, ``"predict"``.

        Raises
        ------
        NotImplementedError
            If stage is not ``"fit"`` or ``"validate"``.
        """
        if stage in ("fit", "validate"):
            if self.train_dataset is not None and self.val_dataset is not None:
                return
            self._setup_train_val()
        else:
            raise NotImplementedError(
                f"Stage '{stage}' is not implemented in MicroSplitDataModule. "
                f"Use stage='fit' or 'validate'."
            )

    def _setup_train_val(self) -> None:
        """Build train and validation ``MicroSplitDataset`` instances."""
        cfg = self.config

        def _build_dataset(data: NDArray, mode_str: str) -> MicroSplitDataset:
            """Build a MicroSplitDataset from an array with the given mode."""
            # Create a mode-specific version of the config.
            if mode_str == "training":
                dataset_config = cfg
            else:
                # Convert to validating mode (uses FixedRandomPatchingConfig).
                dataset_config = cfg.convert_mode("validating")

            return MicroSplitDataset(
                data_config=dataset_config,
                train_data=data,
                multiscale_count=cfg.multiscale_count,
                padding_mode=cfg.padding_mode,
                input_is_sum=cfg.input_is_sum,
                mix_uncorrelated_channels=(
                    cfg.mix_uncorrelated_channels if mode_str == "training" else False
                ),
                uncorrelated_channel_probab=cfg.uncorrelated_channel_probab,
                alpha_range=cfg.alpha_range,
                seed=int(self.rng.integers(1, 2**31)),
            )

        self.train_dataset = _build_dataset(self._train_data, "training")

        val_data = self._val_data if self._val_data is not None else self._train_data
        self.val_dataset = _build_dataset(val_data, "validating")

    def train_dataloader(self) -> DataLoader:
        """Create the training DataLoader.

        Returns
        -------
        DataLoader
            Training dataloader with ``default_collate``.
        """
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=default_collate,
            **self.config.train_dataloader_params,
        )

    def val_dataloader(self) -> DataLoader:
        """Create the validation DataLoader.

        Returns
        -------
        DataLoader
            Validation dataloader with ``default_collate``.
        """
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=default_collate,
            **self.config.val_dataloader_params,
        )
