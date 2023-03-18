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
from collections import OrderedDict
from tqdm import tqdm
from typing import Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data import DataLoader

from .utils import config_validator, getDevice, save_checkpoint
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
        #TODO all initializations of custom classes should be done here
    
    def parse_config(self, cfg_path: str) -> Dict:
        #TODO add smart config validator with blackjack and hookers
        try:
            cfg = yaml.safe_load(Path(cfg_path).open('r'))
        except (FileNotFoundError, yaml.YAMLError):
            #TODO add custom exception for different cases
            raise yaml.YAMLError('Config file not found')
        cfg = config_validator(cfg)
        return cfg
    
    def log_metrics(self):
        if self.cfg.misc.use_wandb:
            try:
                import wandb
                wandb.init(project=self.cfg.experiment_name, config=self.cfg)
                logging.info('using wandb logger')
            except ImportError:
                self.cfg.misc.use_wandb = False
                logging.warning('wandb not installed, using default logger. try pip install wandb')
                return self.log_metrics()
        else:

            logging.info('using default logger')

    def get_model(self):
        return create_model(self.cfg)

    def train(self):
        #General func
        train_loader = self.get_train_dataloader()
        eval_loader = self.get_val_dataloader()
        lr_scheduler = self.get_lr_scheduler()
        optimizer = self.get_optimizer()
        scaler = self.get_grad_scaler()

        try:
            for epoch in range(self.cfg.training.num_epochs):  # loop over the dataset multiple times
                logging.info(f'Starting epoch {epoch}')
                
                train_outputs = self.train_single_epoch(train_loader, optimizer, scaler, self.cfg.training.amp, self.cfg.training.max_grad_norm)
                
                # Perform validation step
                eval_outputs = self.evaluate(eval_loader, self.cfg.evaluation.metric)

                #Add update rule based on type
                lr_scheduler.step()
                save_checkpoint(self.model, 'checkpoint.pth', False)

        except KeyboardInterrupt:
            logging.info('Training interrupted')

    def evaluate(self, eval_loader: torch.utils.data.DataLoader, eval_metric: str):
        self.model.eval()

        metric_func = getattr(src.metrics, eval_metric)
        avg_loss = MetricTracker()
        avg_metric = MetricTracker()

        with torch.no_grad():
            for batch in tqdm(eval_loader):
                outputs = self.model(batch)
                loss = self.get_loss_function(outputs, batch)
                avg_loss.update(loss.item(), batch.shape[0])
                metric = metric_func(outputs, batch)
                avg_metric.update(metric, batch.shape[0])
        return OrderedDict([('loss', avg_loss.avg), (eval_metric, avg_metric.avg)])

    def predict(self, args):
        pass

    def train_single_epoch(self, loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer,
         scaler: torch.cuda.amp.GradScaler, amp: bool, max_grad_norm: Optional[float] = None):
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

        self.model.train()

        for batch in tqdm(loader):
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=amp):
                outputs =self.model(batch)

            loss = self.get_loss_function(outputs, batch)
            scaler.scale(loss).backward()

            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)

            avg_loss.update(loss.item(), batch.shape[0])

            optimizer.step()
        return OrderedDict([('loss', avg_loss.avg)])

    def get_loss_function(self):
        return create_loss_function(self.cfg)

    def get_train_dataloader(self) -> DataLoader:
        dataset = create_dataset(self.cfg, 'training')
        ##TODO add custom collate function and separate dataloader create function
        return DataLoader(
            dataset,
            batch_size=self.cfg['training']['data']['batch_size'],
            num_workers=self.cfg['training']['data']['num_workers'],
        )

    def get_val_dataloader(self) -> DataLoader:
        dataset = create_dataset(self.cfg, 'evaluation')
        return DataLoader(
            dataset,
            batch_size=self.cfg['evaluation']['data']['batch_size'],
            num_workers=self.cfg['evaluation']['data']['num_workers'],
            pin_memory=True,
        )
    
    def get_predict_dataloader(self) -> DataLoader:
        dataset = create_dataset(self.cfg, 'predict')
        return DataLoader(
            dataset,
            batch_size=self.cfg['predict']['data']['batch_size'],
            num_workers=self.cfg['predict']['data']['num_workers'],
            pin_memory=True,
        )

    def get_optimizer(self,
        optimizer_type: str, optimizer_params: Dict
    ) -> Tuple[torch.optim.Optimizer, Dict]:
        """Builds a model based on the model_name or load a checkpoint


        _extended_summary_

        Parameters
        ----------
        model_name : _type_
            _description_
        """
        # assert inspect.get
        #TODO call func from factory
        optimizer = getattr(torch.optim, optimizer_type)
        # Get the list of all possible parameters of the optimizer
        params = _get_params_from_config(optimizer, optimizer_params)
        return optimizer, params


    def get_lr_scheduler(self,
        scheduler_type: str, scheduler_params: Dict
    ) -> Tuple[torch.optim.lr_scheduler._LRScheduler, Dict]:
        """Builds a model based on the model_name or load a checkpoint


        _extended_summary_

        Parameters
        ----------
        model_name : _type_
            _description_
        """

        #TODO call func from factory

        scheduler = getattr(torch.optim.lr_scheduler, scheduler_type)
        params = _get_params_from_config(scheduler, scheduler_params)
        return scheduler, params
    
    def save_checkpoint(self, model, optimizer, scheduler, epoch, loss):
        torch.save(model, os.path.join(self.directory, f'epoch_{epoch}_model.pt'))
    
    def load_checkpoint(self, model, optimizer, scheduler, epoch, loss):
        pass

    def export_model(self, model):
        pass

    def compute_metrics(self, args):
        pass
