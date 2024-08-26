from pydantic import BaseModel, ConfigDict

class TrainingConfig(BaseModel):
    """Configuration for training a VAE model."""
    
    model_config = ConfigDict(
        validate_assignment=True, arbitrary_types_allowed=True, extra="allow"
    )

    batch_size: int = 32 # this is probably in the data config
    """The batch size for training."""
    
    precision: int = 16 # this is used by `Trainer`
    """The precision to use for training."""
    
    grad_clip_norm_value = 0.5 # this is used by `Trainer`
    """The value to use for gradient clipping (see lightning `Trainer`)."""
    
    gradient_clip_algorithm = 'value' # this is used by `Trainer`
    """The algorithm to use for gradient clipping (see lightning `Trainer`)."""
    
    lr_scheduler_patience = 30 # this should go in `optimizer_models/LrSchedulerModel` (?not sure?)
    """The patience for the learning rate scheduler."""
    
    earlystop_patience = 200 # this goes in callbacks params
    """The patience for the learning rate scheduler."""
    
    max_epochs = 10 # this is already there
    """The maximum number of epochs to train for."""
    
    num_workers = 4 # this probably goes in the data config
    """The number of workers to use for data loading."""