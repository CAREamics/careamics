import logging
import random
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .config import ConfigStageEnum, load_configuration
from .dataset import get_tiff_dataset
from .losses import create_loss_function
from .metrics import MetricTracker
from .models import create_model
from .prediction_utils import stitch_prediction
from .utils import (
    denormalize,
    get_device,
    normalize,
    set_logging,
    setup_cudnn_reproducibility,
)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


# TODO: discuss normalization strategies, test running mean and std
class Engine:
    def __init__(self, cfg_path: Union[str, Path]) -> None:
        # set logging
        self.logger = logging.getLogger()
        set_logging(self.logger)

        # load configuration from disk
        self.cfg = load_configuration(cfg_path)

        # create model and loss function
        self.model = create_model(self.cfg)
        self.loss_func = create_loss_function(self.cfg)

        # get GPU or CPU device
        self.device = get_device()

        # seeding
        setup_cudnn_reproducibility(deterministic=True, benchmark=False)
        seed_everything(seed=42)

    def log_metrics(self):
        if self.cfg.misc.use_wandb:
            try:  # TODO Vera will fix this funzione di merda
                import wandb

                wandb.init(project=self.cfg.experiment_name, config=self.cfg)
                self.logger.info("using wandb logger")
            except ImportError:
                self.cfg.misc.use_wandb = False
                self.logger.warning(
                    "wandb not installed, using default logger. try pip install wandb"
                )
                return self.log_metrics()
        else:
            self.logger.info("Using default logger")

    def train(self):
        if self.cfg.training is not None:
            # General func
            train_loader = self.get_dataloader(ConfigStageEnum.TRAINING)

            # TODO what if there are no validation data ? That happens a lot with N2V
            eval_loader = self.get_dataloader(ConfigStageEnum.EVALUATION)

            optimizer, lr_scheduler = self.get_optimizer_and_scheduler()
            scaler = self.get_grad_scaler()
            self.logger.info(
                f"Starting training for {self.cfg.training.num_epochs} epochs"
            )

            val_losses = []
            try:
                for epoch in range(
                    self.cfg.training.num_epochs
                ):  # loop over the dataset multiple times
                    self.logger.info(f"Starting epoch {epoch}")

                    self._train_single_epoch(
                        train_loader,
                        optimizer,
                        scaler,
                        self.cfg.training.amp.use,
                    )

                    # Perform validation step
                    eval_outputs = self.evaluate(eval_loader)
                    self.logger.info(
                        f'Validation loss for epoch {epoch}: {eval_outputs["loss"]}'
                    )
                    # Add update scheduler rule based on type
                    lr_scheduler.step(eval_outputs["loss"])
                    if len(val_losses) == 0 or eval_outputs["loss"] < min(val_losses):
                        self.save_checkpoint(True)
                    else:
                        self.save_checkpoint(False)
                    val_losses.append(eval_outputs["loss"])
                    self.logger.info(
                        f"Saved checkpoint to {self.cfg.working_directory}"
                    )

            except KeyboardInterrupt:
                self.logger.info("Training interrupted")
        else:
            # TODO: instead of error, maybe fail gracefully with a logging/warning to users
            raise ValueError("Missing training entry in configuration file.")

    def _train_single_epoch(
        self,
        loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scaler: torch.cuda.amp.GradScaler,
        amp: bool,
    ):
        """_summary_.

        _extended_summary_

        Parameters
        ----------
        model : _type_
            _description_
        loader : _type_
            _description_
        """
        avg_loss = MetricTracker()
        self.model.to(self.device)
        self.model.train()

        for batch, *auxillary in tqdm(loader):
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=amp):
                outputs = self.model(batch.to(self.device))

            loss = self.loss_func(outputs, *auxillary, self.device)
            scaler.scale(loss).backward()

            avg_loss.update(loss.item(), batch.shape[0])

            optimizer.step()
        return {"loss": avg_loss.avg}

    def evaluate(self, eval_loader: torch.utils.data.DataLoader):
        self.model.eval()
        avg_loss = MetricTracker()

        with torch.no_grad():
            for patch, *auxillary in tqdm(eval_loader):
                outputs = self.model(patch.to(self.device))
                loss = self.loss_func(outputs, *auxillary, self.device)
                avg_loss.update(loss.item(), patch.shape[0])

        return {"loss": avg_loss.avg}

    def predict(self, external_input: Optional[np.ndarray] = None):
        self.model.to(self.device)
        self.model.eval()
        # TODO external input shape should either be compatible with the model or tiled. Add checks and raise errors
        if not (self.mean and self.std):
            _, self.mean, self.std = self._get_train_dataloader()

        # TODO external input shape should either be compatible with the model or tiled. Add checks and raise errors
        # TODO check, calculate mean and std on all data not only train
        pred_loader, stitch = self.get_predict_dataloader(
            external_input=external_input,
            mean=self.cfg.data.mean,
            std=self.cfg.data.mean,
        )

        tiles = []
        prediction = []
        if external_input is not None:
            logging.info("Starting prediction on external input")
        if stitch:
            self.logger.info("Starting tiled prediction")
        else:
            self.logger.info("Starting prediction on whole sample")

        with torch.no_grad():
            # TODO reset iterator for every sample ?
            # TODO tiled prediction slow af, profile and optimize
            for _, (tile, *auxillary) in tqdm(enumerate(pred_loader)):
                if auxillary:
                    (
                        last_tile,
                        sample_shape,
                        overlap_crop_coords,
                        stitch_coords,
                    ) = auxillary

                outputs = self.model(tile.to(self.device))
                outputs = denormalize(
                    outputs, pred_loader.dataset.mean, pred_loader.dataset.std
                )

                if stitch:
                    # TODO: can the code be expanded to have the squeezing done
                    # not in one liners, also check the detach() and numpy()

                    # Crop predited tile according to overlap coordinates
                    predicted_tile = outputs.squeeze()[
                        (
                            ...,
                            *[
                                slice(c.squeeze()[0], c.squeeze()[1])
                                for c in list(overlap_crop_coords)
                            ],
                        )
                    ]
                    # TODO: removing ellipsis works for 3.11
                    """ 3.11 syntax
                    predicted_tile = outputs.squeeze()[
                        *[
                            slice(c.squeeze()[0], c.squeeze()[1])
                            for c in list(overlap_crop_coords)
                        ],
                    ]
                    """
                    tiles.append(
                        (
                            predicted_tile.cpu().numpy(),
                            [c.squeeze().numpy() for c in stitch_coords],
                        )
                    )
                    # check if sample is finished
                    if last_tile:
                        # Stitch tiles together
                        predicted_sample = stitch_prediction(tiles, sample_shape)
                        prediction.append(predicted_sample)
                else:
                    prediction.append(outputs.detach().cpu().numpy().squeeze())

        self.logger.info(f"Predicted {len(prediction)} samples")
        return np.stack(prediction)

    # TODO: add custom collate function and separate dataloader create function, sampler?
    def get_dataloader(self, stage: ConfigStageEnum) -> DataLoader:
        dataset = get_tiff_dataset(self.cfg, stage)
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.training.batch_size,
            num_workers=self.cfg.training.num_workers,
            pin_memory=True,
        )
        return dataloader

    def get_predict_dataloader(
        self,
        external_input: Optional[np.ndarray] = None,
    ) -> Tuple[DataLoader, bool]:
        # TODO add description
        # TODO mypy does not take into account "is not None", we need to find a workaround
        if external_input is not None:
            normalized_input = normalize(external_input, self.mean, self.std)
            normalized_input = normalized_input.astype(np.float32)

            dataset = TensorDataset(torch.from_numpy(normalized_input))
            stitch = False
        else:
            dataset = get_tiff_dataset(self.cfg, ConfigStageEnum.PREDICTION)
            stitch = (
                hasattr(dataset, "patch_generator")
                and dataset.patch_generator is not None
            )
        return (
            # TODO there is not batch_size and num_workers in prediction
            DataLoader(
                dataset,
                batch_size=1,  # self.cfg.prediction.data.batch_size,
                num_workers=0,  # self.cfg.prediction.data.num_workers,
                pin_memory=True,
            ),
            stitch,
        )

    def get_optimizer_and_scheduler(
        self,
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        """Builds a model based on the model_name or load a checkpoint.

        _extended_summary_

        Parameters
        ----------
        model_name : _type_
            _description_
        """
        # assert inspect.get
        # TODO call func from factory
        if self.cfg.training is not None:
            # retrieve optimizer name and parameters from config
            optimizer_name = self.cfg.training.optimizer.name
            optimizer_params = self.cfg.training.optimizer.parameters

            # then instantiate it
            optimizer_func = getattr(torch.optim, optimizer_name)
            optimizer = optimizer_func(self.model.parameters(), **optimizer_params)

            # same for learning rate scheduler
            scheduler_name = self.cfg.training.lr_scheduler.name
            scheduler_params = self.cfg.training.lr_scheduler.parameters
            scheduler_func = getattr(torch.optim.lr_scheduler, scheduler_name)
            scheduler = scheduler_func(optimizer, **scheduler_params)

            return optimizer, scheduler
        else:
            raise ValueError("Missing training entry in configuration file.")

    def get_grad_scaler(self) -> torch.cuda.amp.GradScaler:
        if self.cfg.training is not None:
            use = self.cfg.training.amp.use
            scaling = self.cfg.training.amp.init_scale
            return torch.cuda.amp.GradScaler(init_scale=scaling, enabled=use)
        else:
            raise ValueError("Missing training entry in configuration file.")

    def save_checkpoint(self, save_best):
        """Save the model to a checkpoint file."""
        name = (
            f"{self.cfg.experiment_name}_best.pth"
            if save_best
            else f"{self.cfg.experiment_name}_latest.pth"
        )
        workdir = self.cfg.working_directory
        workdir.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), workdir / name)
