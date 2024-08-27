import os
import json
import socket
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Literal, Union

import git
import ml_collections
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pydantic import BaseModel, ConfigDict

from careamics.config import VAEAlgorithmConfig
from careamics.config.architectures import LVAEModel
from careamics.config.likelihood_model import (
    GaussianLikelihoodConfig,
    NMLikelihoodConfig,
)
from careamics.config.nm_model import GaussianMixtureNMConfig, MultiChannelNMConfig
from careamics.config.optimizer_models import OptimizerModel, LrSchedulerModel
from careamics.lightning import VAEModule
from careamics.models.lvae.noise_models import noise_model_factory
from careamics.lvae_training.data_modules import MultiChDloader, LCMultiChDloader
from careamics.lvae_training.data_utils import DataType, DataSplitType
from careamics.lvae_training.train_utils import get_new_model_version


# --- Custom parameters
img_size: int = 64
"""Spatial size of the input image."""
target_channels: int = 2
"""Number of channels in the target image."""
multiscale_count: int = 1
"""The number of LC inputs plus one (the actual input)."""
predict_logvar: Optional[Literal["pixelwise"]] = "pixelwise"
"""Whether to compute also the log-variance as LVAE output."""
loss_type: Optional[Literal["musplit", "denoisplit", "denoisplit_musplit"]] = "musplit"
"""The type of reconstruction loss (i.e., likelihood) to use."""

# --- Training parameters
# TODO: replace with PR #225
class TrainingConfig(BaseModel):
    """Configuration for training a VAE model."""
    
    model_config = ConfigDict(
        validate_assignment=True, arbitrary_types_allowed=True, extra="allow"
    )

    batch_size: int = 32
    """The batch size for training."""
    precision: int = 16
    """The precision to use for training."""
    lr: float = 1e-3
    """The learning rate for training."""
    lr_scheduler_patience: int = 30
    """The patience for the learning rate scheduler."""
    earlystop_patience: int = 200
    """The patience for the learning rate scheduler."""
    max_epochs: int = 400
    """The maximum number of epochs to train for."""
    num_workers: int = 4
    """The number of workers to use for data loading."""
    grad_clip_norm_value: int = 0.5
    """The value to use for gradient clipping (see lightning `Trainer`)."""
    gradient_clip_algorithm: int = 'value'
    """The algorithm to use for gradient clipping (see lightning `Trainer`)."""
    
train_config = TrainingConfig()


### --- Data parameters
def get_data_config(
):
    data_config = ml_collections.ConfigDict()
    data_config.image_size = img_size
    data_config.target_channels = target_channels
    data_config.multiscale_lowres_count = multiscale_count #wrong -> 3
    data_config.data_type = DataType.BioSR_MRC
    data_config.ch1_fname = "ER/GT_all.mrc"
    data_config.ch2_fname = "CCPs/GT_all.mrc" 
    data_config.poisson_noise_factor = -1
    data_config.enable_gaussian_noise = True
    data_config.synthetic_gaussian_scale = 5100
    data_config.input_has_dependant_noise = True
    return data_config


### --- Functions to create datasets and model
def create_dataset(
    config: ml_collections.ConfigDict,
    datadir: str,
    eval_datasplit_type = DataSplitType.Val,
    skip_train_dataset = False,
    kwargs_dict = None,
) -> tuple[Dataset, Dataset, tuple[float, float]]:
    if kwargs_dict is None:
        kwargs_dict = {}

    datapath = datadir

    # Hard-coded parameters (used to be in the config file)
    normalized_input = True
    use_one_mu_std = True
    train_aug_rotate = False
    enable_random_cropping = True
    lowres_supervision = False

    # 1) Data loader for Lateral Contextualization
    if config.multiscale_lowres_count > 1:
        # Get padding attributes
        if "padding_kwargs" not in kwargs_dict:
            padding_kwargs = {}
            if "padding_mode" in config and config.padding_mode is not None:
                padding_kwargs["mode"] = config.padding_mode
            else:
                padding_kwargs["mode"] = "reflect" 
            if "padding_value" in config and config.padding_value is not None:
                padding_kwargs["constant_values"] = config.padding_value
            else:
                padding_kwargs["constant_values"] = None 
        else:
            padding_kwargs = kwargs_dict.pop("padding_kwargs")

        train_data = (
            None
            if skip_train_dataset
            else LCMultiChDloader(
                config,
                datapath,
                datasplit_type=DataSplitType.Train,
                val_fraction=0.1,
                test_fraction=0.1,
                normalized_input=normalized_input,
                use_one_mu_std=use_one_mu_std,
                enable_rotation_aug=train_aug_rotate,
                enable_random_cropping=enable_random_cropping,
                num_scales=config.multiscale_lowres_count,
                lowres_supervision=lowres_supervision,
                padding_kwargs=padding_kwargs,
                **kwargs_dict,
                allow_generation=True,
            )
        )
        max_val = train_data.get_max_val()

        val_data = LCMultiChDloader(
            config,
            datapath,
            datasplit_type=eval_datasplit_type,
            val_fraction=0.1,
            test_fraction=0.1,
            normalized_input=normalized_input,
            use_one_mu_std=use_one_mu_std,
            enable_rotation_aug=False,  # No rotation aug on validation
            enable_random_cropping=False,
            # No random cropping on validation. Validation is evaluated on determistic grids
            num_scales=config.multiscale_lowres_count,
            lowres_supervision=lowres_supervision,
            padding_kwargs=padding_kwargs,
            allow_generation=False,
            **kwargs_dict,
            max_val=max_val,
        )
    # 2) Vanilla data loader
    else:
        train_data_kwargs = {"allow_generation": True, **kwargs_dict}
        val_data_kwargs = {"allow_generation": False, **kwargs_dict}

        train_data_kwargs["enable_random_cropping"] = enable_random_cropping
        val_data_kwargs["enable_random_cropping"] = False

        train_data = (
            None
            if skip_train_dataset
            else MultiChDloader(
                data_config=config,
                fpath=datapath,
                datasplit_type=DataSplitType.Train,
                val_fraction=0.1,
                test_fraction=0.1,
                normalized_input=normalized_input,
                use_one_mu_std=use_one_mu_std,
                enable_rotation_aug=train_aug_rotate,
                **train_data_kwargs,
            )
        )

        max_val = train_data.get_max_val()
        val_data = MultiChDloader(
            data_config=config,
            fpath=datapath,
            datasplit_type=eval_datasplit_type,
            val_fraction=0.1,
            test_fraction=0.1,
            normalized_input=normalized_input,
            use_one_mu_std=use_one_mu_std,
            enable_rotation_aug=False,  # No rotation aug on validation
            max_val=max_val,
            **val_data_kwargs,
        )
    
    mean_val, std_val = train_data.compute_mean_std()
    train_data.set_mean_std(mean_val, std_val)
    val_data.set_mean_std(mean_val, std_val)
    data_stats = train_data.get_mean_std()
    
    # NOTE: "input" mean & std are computed over the entire dataset and repeated for each channel.
    # On the contrary, "target" mean & std are computed separately for each channel.
    
    return train_data, val_data, data_stats


def create_split_lightning_model(
    algorithm: str,
    loss_type: str,
    img_size: int = 64,
    multiscale_count: int = 1,
    predict_logvar: Optional[Literal["pixelwise"]] = None,
    target_ch: int = 1,
    NM_path: Optional[Path] = None,
    training_config: TrainingConfig = TrainingConfig() 
) -> VAEModule:
    """Instantiate the muSplit lightining model."""
    lvae_config = LVAEModel(
        architecture="LVAE",
        input_shape=img_size,
        multiscale_count=multiscale_count,
        z_dims=[128, 128, 128, 128],
        output_channels=target_ch,
        predict_logvar=predict_logvar,
        analytical_kl=False, 
    )

    # gaussian likelihood
    if loss_type in ["musplit", "denoisplit_musplit"]:
        gaussian_lik_config = GaussianLikelihoodConfig(
            predict_logvar=predict_logvar,
            logvar_lowerbound=0.0,
        )
    else:
        gaussian_lik_config = None
    # noise model likelihood
    if loss_type in ["denoisplit", "denoisplit_musplit"]:
        assert NM_path is not None, "A path to a pre-trained noise model is required."
        
        gmm = GaussianMixtureNMConfig(
            model_type="GaussianMixtureNoiseModel",
            path=NM_path,
        )
        noise_model_config = MultiChannelNMConfig(noise_models=[gmm] * target_ch)
        nm = noise_model_factory(noise_model_config)
        nm_lik_config = NMLikelihoodConfig(
            noise_model=nm
            # TODO: pass data_mean and data_std from dataset
        )
    else:
        noise_model_config = None
        nm_lik_config = None
    
    opt_config = OptimizerModel(
        name="Adamax",
        parameters={
            "lr": training_config.lr,
            "weight_decay": 0,   
        },
    )
    lr_scheduler_config = LrSchedulerModel(
        name="ReduceLROnPlateau",
        parameters={
            "mode": "min",
            "factor": 0.5,
            "patience": training_config.lr_scheduler_patience,
            "verbose": True,
            "min_lr": 1e-12,
        },
    )

    vae_config = VAEAlgorithmConfig(
        algorithm_type="vae",
        algorithm=algorithm,
        loss=loss_type,
        model=lvae_config,
        gaussian_likelihood_model=gaussian_lik_config,
        noise_model=noise_model_config,
        noise_model_likelihood_model=nm_lik_config,
        optimizer=opt_config,
        lr_scheduler=lr_scheduler_config,
    )

    return VAEModule(algorithm_config=vae_config)

# --- Utils
def get_new_model_version(model_dir: Union[Path, str]) -> int:
    """Create a unique version ID for a new model run."""
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

def get_workdir(
    root_dir: str,
    model_name: str,
) -> tuple[Path, Path]:
    """Get the workdir for the current model.
    
    It has the following structure: "root_dir/YYMM/model_name/version"
    """
    rel_path = datetime.now().strftime("%y%m")
    cur_workdir = os.path.join(root_dir, rel_path)
    Path(cur_workdir).mkdir(exist_ok=True)

    rel_path = os.path.join(rel_path, model_name)
    cur_workdir = os.path.join(root_dir, rel_path)
    Path(cur_workdir).mkdir(exist_ok=True)

    rel_path = os.path.join(rel_path, get_new_model_version(cur_workdir))
    cur_workdir = os.path.join(root_dir, rel_path)
    try:
        Path(cur_workdir).mkdir(exist_ok=False)
    except FileExistsError:
        print(
            f"Workdir {cur_workdir} already exists."
        )
    return cur_workdir, rel_path

def get_git_status() -> dict[Any]:
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    repo = git.Repo(curr_dir, search_parent_directories=True)
    git_config = {}
    git_config["changedFiles"] = [item.a_path for item in repo.index.diff(None)]
    git_config["branch"] = repo.active_branch.name
    git_config["untracked_files"] = repo.untracked_files
    git_config["latest_commit"] = repo.head.object.hexsha
    return git_config


def main():
    
    training_config = TrainingConfig() 
    
    # --- Get dloader
    train_dset, val_dset, data_stats = create_dataset(
        config=get_data_config(),
        datadir="/group/jug/federico/careamics_training/data/BioSR",
        eval_datasplit_type=DataSplitType.Val,
        skip_train_dataset=False,
        kwargs_dict=None,
    )
    train_dloader = DataLoader(
        train_dset, 
        batch_size=train_config.batch_size, 
        num_workers=train_config.num_workers, 
        shuffle=True
    )
    val_dloader = DataLoader(
        val_dset, 
        batch_size=train_config.batch_size, 
        num_workers=train_config.num_workers, 
        shuffle=False
    )
    
    algo = "musplit" if loss_type == "musplit" else "denoisplit"
    lightning_model = create_split_lightning_model(
        algorithm=algo,
        loss_type=loss_type,
        img_size=img_size,
        multiscale_count=multiscale_count,
        predict_logvar=predict_logvar,
        target_ch=target_channels,
        NM_path=None,
        training_config=train_config
    )
    
    ROOT_DIR = "/group/jug/federico/careamics_training/refac_v2/"
    workdir, exp_tag = get_workdir(ROOT_DIR, "musplit_no_LC")
    print(f"Current workdir: {workdir}")
    
    
    # Define the logger
    custom_logger = WandbLogger(
        name=os.path.join(socket.gethostname(), exp_tag),
        save_dir=workdir,
        project="_".join(("careamics", algo, "LC", str(multiscale_count))),
    )

    
    # Define callbacks (e.g., ModelCheckpoint, EarlyStopping, etc.)
    custom_callbacks = [
        EarlyStopping(
            monitor="val_loss",
            min_delta=1e-6,
            patience=train_config.earlystop_patience,
            mode="min",
            verbose=True,        
        ), 
        ModelCheckpoint(
            dirpath=os.path.join(workdir, "checkpoints"),
            filename="best-{epoch}",
            monitor="val_loss",
            save_top_k=1,
            save_last=True,
            mode="min",    
        ),
        LearningRateMonitor(logging_interval="epoch")
    ]
    
    # Save configs and git status (for debugging)
    with open(os.path.join(workdir, "algorithm_config.json"), "w") as f:
        f.write(lightning_model.algorithm_config.model_dump_json())

    with open(os.path.join(workdir, "training_config.json"), "w") as f:
        f.write(train_config.model_dump_json())
        
    with open(os.path.join(workdir, "git_config.json"), "w") as f:
        json.dump(get_git_status(), f)
        
    # Train the model
    trainer = Trainer(
        max_epochs=train_config.max_epochs,
        accelerator="gpu",
        enable_progress_bar=True,
        logger=custom_logger,
        callbacks=custom_callbacks,
        precision=train_config.precision,
        gradient_clip_val=train_config.grad_clip_norm_value, # only works with `accelerator="gpu"`
        gradient_clip_algorithm=train_config.gradient_clip_algorithm,
    )
    trainer.fit(
        model=lightning_model,
        train_dataloaders=train_dloader,
        val_dataloaders=val_dloader,
    )
  
  
if __name__ == "__main__":
    main()    
