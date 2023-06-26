from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict
import logging

import yaml
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .metrics import MetricTracker
from .dataloader_utils import (
    create_dataset,
)
from .losses import create_loss_function
from .config import Configuration, load_configuration, get_parameters
from .utils import set_logging, get_device, normalize, denormalize
from .prediction_utils import stitch_prediction
from .models import create_model


class Engine(ABC):
    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def get_train_dataloader(self):
        pass

    @abstractmethod
    def get_predict_dataloader(self):
        pass

    @abstractmethod
    def train(self, args):
        pass

    @abstractmethod
    def train_single_epoch(self, args):
        pass

    # @abstractmethod
    # def predict(self, args):
    #     pass


class UnsupervisedEngine(Engine):
    def __init__(self, cfg_path: str) -> None:
        self.logger = logging.getLogger()
        set_logging(self.logger)
        self.cfg = self.parse_config(cfg_path)
        self.model = self.get_model()
        self.loss_func = self.get_loss_function()
        self.mean = None
        self.std = None
        self.device = get_device()

    def parse_config(self, cfg_path: str) -> Dict:
        try:
            cfg = load_configuration(cfg_path)
        except (FileNotFoundError, yaml.YAMLError):
            raise yaml.YAMLError(f"Config file not found in {cfg_path}")
        cfg = Configuration(**cfg)
        return cfg

    def log_metrics(self):
        if self.cfg.misc.use_wandb:
            try:  # TODO test wandb. add functionality
                import wandb

                wandb.init(project=self.cfg.run_params.experiment_name, config=self.cfg)
                self.logger.info("using wandb logger")
            except ImportError:
                self.cfg.misc.use_wandb = False
                self.logger.warning(
                    "wandb not installed, using default logger. try pip install wandb"
                )
                return self.log_metrics()
        else:
            self.logger.info("Using default logger")

    def get_model(self):
        return create_model(self.cfg)

    def train(self):
        # General func
        train_loader, self.mean, self.std = self.get_train_dataloader()
        eval_loader = self.get_val_dataloader()
        eval_loader.dataset.set_normalization(self.mean, self.std)
        optimizer, lr_scheduler = self.get_optimizer_and_scheduler()
        scaler = self.get_grad_scaler()
        self.logger.info(f"Starting training for {self.cfg.training.num_epochs} epochs")

        val_losses = []  # TODO reimplement this ugly shit
        try:
            for epoch in range(
                self.cfg.training.num_epochs
            ):  # loop over the dataset multiple times
                self.logger.info(f"Starting epoch {epoch}")

                train_outputs = self.train_single_epoch(
                    train_loader,
                    optimizer,
                    scaler,
                    self.cfg.training.amp.use,
                    self.cfg.training.max_grad_norm,
                )

                # Perform validation step
                eval_outputs = self.evaluate(eval_loader, self.cfg.evaluation.metric)
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
                self.logger.info(f"Saved checkpoint to {self.cfg.run_params.workdir}")

        except KeyboardInterrupt:
            self.logger.info("Training interrupted")

    def evaluate(self, eval_loader: torch.utils.data.DataLoader, eval_metric: str):
        self.model.eval()
        avg_loss = MetricTracker()
        avg_loss.reset()

        with torch.no_grad():
            for patch, *auxillary in tqdm(eval_loader):
                outputs = self.model(patch.to(self.device))
                loss = self.loss_func(outputs, *auxillary, self.device)
                avg_loss.update(loss.item(), patch.shape[0])

        return {"loss": avg_loss.avg}

    def predict(self, args):
        # TODO predict whole folder. Need filenames from DL, indices of samples in each file

        self.model.to(self.device)
        self.model.eval()
        if not (self.mean and self.std):
            _, self.mean, self.std = self.get_train_dataloader()
        pred_loader = self.get_predict_dataloader(
            ext_input=ext_input
        )  # TODO check, calculate mean and std on all data not only train
        self.stitch = (
            hasattr(pred_loader.dataset, "patch_generator")
            and pred_loader.dataset.patch_generator is not None
        )
        if self.stitch:
            self.logger.info("Starting tiled prediction")
        else:
            self.logger.info("Starting prediction on whole sample")
        with torch.no_grad():
            for idx, (tile, *auxillary) in tqdm(enumerate(pred_loader)):
                # TODO define all predict/train funcs in separate modules
                if auxillary:
                    (
                        sample_idx,
                        overlap_crop_coords,
                        stitch_coords,
                    ) = auxillary
            # TODO get filename from DL, get index treshold marking end of file
        pass

    def predict_from_memory(self):
        # TODO predict on externally passed array
        pass

    def predict_from_disk(self):
        # TODO predict on externally passed path
        pass

    def predict_single_sample(self, ext_input: np.ndarray = None):
        self.model.to(self.device)
        self.model.eval()
        if not (self.mean and self.std):
            _, self.mean, self.std = self.get_train_dataloader()
        pred_loader = self.get_predict_dataloader(
            ext_input=ext_input
        )  # TODO check, calculate mean and std on all data not only train
        self.stitch = (
            hasattr(pred_loader.dataset, "patch_generator")
            and pred_loader.dataset.patch_generator is not None
        )
        avg_metric = MetricTracker()
        tiles = []
        if self.stitch:
            self.logger.info("Starting tiled prediction")
        else:
            self.logger.info("Starting prediction on whole sample")
        with torch.no_grad():
            for idx, (tile, *auxillary) in tqdm(enumerate(pred_loader)):
                # TODO define all predict/train funcs in separate modules
                if auxillary:
                    (
                        sample_idx,
                        sample_shape,
                        overlap_crop_coords,
                        stitch_coords,
                    ) = auxillary

                # TODO get sample, iterate over tiles of 1 sample
                outputs = self.model(
                    tile.to(self.device)
                )  # Why batch dimension is not added by dl ?
                outputs = denormalize(outputs, self.mean, self.std)
                if self.stitch:
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
                    # TODO Do we need to save tiles separately ?
                    tiles.append(
                        (
                            predicted_tile.cpu().numpy(),
                            [c.squeeze().numpy() for c in stitch_coords],
                        )
                    )
                else:
                    tiles.append((outputs.detach().cpu().numpy(), None))

        # Stitch tiles together
        if self.stitch:
            self.logger.info("Stitching tiles together")
            pred = stitch_prediction(tiles, sample_shape)
            self.logger.info("Prediction finished")
            return pred, tiles
        else:
            self.logger.info("Prediction finished")
            return None, tiles

    def train_single_epoch(
        self,
        loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scaler: torch.cuda.amp.GradScaler,
        amp: bool,
        max_grad_norm: Optional[float] = None,
    ):
        """_summary_

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

        for patch, *auxillary in tqdm(loader):
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=amp):
                outputs = self.model(patch.to(self.device))
            loss = self.loss_func(outputs, *auxillary, self.device)
            scaler.scale(loss).backward()

            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=max_grad_norm
                )
            avg_loss.update(loss.item(), patch.shape[0])

            optimizer.step()
        return {"loss": avg_loss.avg}

    def get_loss_function(self):
        return create_loss_function(self.cfg)

    def get_train_dataloader(self) -> DataLoader:
        dataset = create_dataset(self.cfg, "training")
        ##TODO add custom collate function and separate dataloader create function, sampler?
        if not self.cfg.training.running_stats:
            self.logger.info(f"Calculating mean/std of the data")
            dataset.calculate_stats()
        else:
            self.logger.info(f"Using running average of mean/std")
        return (
            DataLoader(
                dataset,
                batch_size=self.cfg.training.data.batch_size,
                num_workers=self.cfg.training.data.num_workers,
            ),
            dataset.mean,
            dataset.std,
        )

    # TODO merge into single dataloader func ?
    def get_val_dataloader(self) -> DataLoader:
        dataset = create_dataset(self.cfg, "evaluation")
        return DataLoader(
            dataset,
            batch_size=self.cfg.evaluation.data.batch_size,
            num_workers=self.cfg.evaluation.data.num_workers,
            pin_memory=True,
        )

    def get_predict_dataloader(self, ext_input: np.ndarray = None) -> DataLoader:
        # TODO add description
        if ext_input is not None:
            ext_input = normalize(ext_input, self.mean, self.std)
            dataset = torch.utils.data.TensorDataset(
                torch.from_numpy(ext_input.astype(np.float32))
            )
        else:
            dataset = create_dataset(self.cfg, "prediction")
            dataset.set_normalization(self.mean, self.std)
        return DataLoader(
            dataset,
            batch_size=self.cfg.prediction.data.batch_size,
            num_workers=self.cfg.prediction.data.num_workers,
            pin_memory=True,
        )

    def get_optimizer_and_scheduler(
        self,
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        """Builds a model based on the model_name or load a checkpoint


        _extended_summary_

        Parameters
        ----------
        model_name : _type_
            _description_
        """
        # assert inspect.get
        # TODO call func from factory
        optimizer_name = self.cfg.training.optimizer.name
        optimizer_params = self.cfg.training.optimizer.parameters
        optimizer_func = getattr(torch.optim, optimizer_name)
        # Get the list of all possible parameters of the optimizer
        optim_params = get_parameters(optimizer_func, optimizer_params)
        # TODO add support for different learning rates for different layers
        optimizer = optimizer_func(self.model.parameters(), **optim_params)

        scheduler_name = self.cfg.training.lr_scheduler.name
        scheduler_params = self.cfg.training.lr_scheduler.parameters
        scheduler_func = getattr(torch.optim.lr_scheduler, scheduler_name)
        scheduler_params = get_parameters(scheduler_func, scheduler_params)
        scheduler = scheduler_func(optimizer, **scheduler_params)
        return optimizer, scheduler

    def get_grad_scaler(self) -> torch.cuda.amp.GradScaler:
        use = self.cfg.training.amp.use
        scaling = self.cfg.training.amp.init_scale
        return torch.cuda.amp.GradScaler(init_scale=scaling, enabled=use)

    def save_checkpoint(self, save_best):
        """Save the model to a checkpoint file."""
        name = (
            f"{self.cfg.run_params.experiment_name}_best.pth"
            if save_best
            else f"{self.cfg.run_params.experiment_name}_latest.pth"
        )
        Path(self.cfg.run_params.workdir).mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), Path(self.cfg.run_params.workdir) / name)

    def export_model(self, model):
        pass

    def compute_metrics(self, args):
        pass
