import os
import sys
import yaml
import logging
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from abc import ABC, abstractmethod
from tqdm import tqdm
from typing import Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data import DataLoader

# TODO sort imports here
from . import *

from .metrics import MetricTracker
from .factory import (
    _get_params_from_config,
    _get_params_from_module,
    create_model,
    create_dataset,
    create_loss_function,
    create_grad_scaler,
    create_optimizer,
    create_lr_scheduler,
)

# TODO do something with imports, it's a mess. either all from n2v init, or all separately


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

    @abstractmethod
    def predict(self, args):
        pass


class UnsupervisedEngine(Engine):
    def __init__(self, cfg_path: str) -> None:
        self.cfg = self.parse_config(cfg_path)
        self.model = self.get_model()
        self.loss_func = self.get_loss_function()
        self.device = get_device()
        # TODO all initializations of custom classes should be done here

    def parse_config(self, cfg_path: str) -> Dict:
        try:
            cfg = config_loader(cfg_path)
        except (FileNotFoundError, yaml.YAMLError):
            # TODO add custom exception for different cases
            raise yaml.YAMLError("Config file not found")
        cfg = ConfigValidator(**cfg)
        return cfg

    def log_metrics(self):
        if self.cfg.misc.use_wandb:
            try:
                import wandb

                wandb.init(project=self.cfg.experiment_name, config=self.cfg)
                logging.info("using wandb logger")
            except ImportError:
                self.cfg.misc.use_wandb = False
                logging.warning(
                    "wandb not installed, using default logger. try pip install wandb"
                )
                return self.log_metrics()
        else:
            logging.info("using default logger")

    def get_model(self):
        return create_model(self.cfg)

    def train(self):
        # TODO move to main
        set_logging()

        # General func
        train_loader = self.get_train_dataloader()
        eval_loader = self.get_val_dataloader()
        optimizer, lr_scheduler = self.get_optimizer_and_scheduler()
        scaler = self.get_grad_scaler()

        logging.info(
            f'Starting training for {self.cfg.training.num_epochs} epochs'
        )

        try:
            for epoch in range(
                self.cfg.training.num_epochs
            ):  # loop over the dataset multiple times
                logging.info(f"Starting epoch {epoch}")

                train_outputs = self.train_single_epoch(
                    train_loader,
                    optimizer,
                    scaler,
                    self.cfg.training.amp.toggle,
                    self.cfg.training.max_grad_norm,
                )

                # Perform validation step
                eval_outputs = self.evaluate(
                    eval_loader, self.cfg.evaluation.metric
                )

                # Add update scheduler rule based on type
                lr_scheduler.step(eval_outputs["loss"])
                #TODO implement checkpoint naming
                save_checkpoint(self.model, "checkpoint.pth", False)

        except KeyboardInterrupt:
            logging.info("Training interrupted")

    def evaluate(self, eval_loader: torch.utils.data.DataLoader, eval_metric: str):
        self.model.eval()
        #TODO Isnt supposed to be called without train ?
        avg_loss = MetricTracker()

        with torch.no_grad():
            for image, *auxillary in tqdm(eval_loader):
                outputs = self.model(image.to(self.device))
                loss = self.loss_func(
                    outputs, *auxillary, self.device, 1
                )
                avg_loss.update(loss.item(), image.shape[0])
        return {"loss": avg_loss.avg}

    def predict(self):
        self.model.to(self.device)
        self.model.eval()

        pred_loader = self.get_predict_dataloader()
        avg_metric = MetricTracker()
        inputs = []
        preds = []
        with torch.no_grad():
            for image, *auxillary in tqdm(pred_loader):
                outputs = self.model(image.to(self.device))
                #TODO tile predict from aux 
                #TODO calc function placement ?

                tile_coords, max_shapes, overlap = auxillary
                coords = calculate_stitching_coords(tile_coords, max_shapes, overlap)
                
                pred.append(tile[(*[c for c in coords], ...)]) #TODO add proper last tile coord ! Should be list !)
                inputs.append(image)
                preds.append(outputs)
        
        return inputs, preds

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

        for image, *auxillary in tqdm(loader):
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=amp):
                # TODO add normalization
                outputs = self.model(image.to(self.device))
            # TODO std !!
            loss = self.loss_func(
                outputs, *auxillary, self.device, 1
            )
            scaler.scale(loss).backward()

            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=max_grad_norm
                )
            # TODO fix batches naming
            avg_loss.update(loss.item(), image.shape[0])

            optimizer.step()
        return {"loss": avg_loss.avg}

    def get_loss_function(self):
        return create_loss_function(self.cfg)

    def get_train_dataloader(self) -> DataLoader:
        dataset = create_dataset(self.cfg, "training")
        ##TODO add custom collate function and separate dataloader create function, sampler?
        return DataLoader(
            dataset,
            batch_size=self.cfg.training.data.batch_size,
            num_workers=self.cfg.training.data.num_workers,
        )
    #TODO merge into single dataloader func ?
    def get_val_dataloader(self) -> DataLoader:
        dataset = create_dataset(self.cfg, "evaluation")
        return DataLoader(
            dataset,
            batch_size=self.cfg.evaluation.data.batch_size, 
            num_workers=self.cfg.evaluation.data.num_workers,
            pin_memory=True,
        )

    def get_predict_dataloader(self) -> DataLoader:
        dataset = create_dataset(self.cfg, "prediction")
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
        optim_params = _get_params_from_config(optimizer_func, optimizer_params)
        # TODO add support for different learning rates for different layers
        optimizer = optimizer_func(self.model.parameters(), **optim_params)

        scheduler_name = self.cfg.training.lr_scheduler.name
        scheduler_params = self.cfg.training.lr_scheduler.parameters
        scheduler_func = getattr(torch.optim.lr_scheduler, scheduler_name)
        scheduler_params = _get_params_from_config(scheduler_func, scheduler_params)
        scheduler = scheduler_func(optimizer, **scheduler_params)
        return optimizer, scheduler

    def get_grad_scaler(self) -> torch.cuda.amp.GradScaler:
        toggle = self.cfg.training.amp.toggle
        scaling = self.cfg.training.amp.init_scale
        return torch.cuda.amp.GradScaler(init_scale=scaling, enabled=toggle)

    def save_checkpoint(self, model, optimizer, scheduler, epoch, loss):
        torch.save(model, os.path.join(self.directory, f"epoch_{epoch}_model.pt"))

    def load_checkpoint(self, model, optimizer, scheduler, epoch, loss):
        pass

    def export_model(self, model):
        pass

    def compute_metrics(self, args):
        pass
