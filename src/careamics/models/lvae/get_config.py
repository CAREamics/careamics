"""
Here there are functions to define a config file.
"""
import os
import ml_collections

def _init_config():
    """
    Create a default config object with all the required fields.
    """
    config = ml_collections.ConfigDict()

    config.data = ml_collections.ConfigDict()

    config.model = ml_collections.ConfigDict()

    config.loss = ml_collections.ConfigDict()

    config.training = ml_collections.ConfigDict()

    config.workdir = os.getcwd()
    config.datadir = ''
    return config
    
def get_config():
    config = _init_config()
    
    data = config.data
    data.image_size = 128
    data.multiscale_lowres_count = None
    
    model = config.model
    model.z_dims = [128, 128, 128, 128]
    model.n_filters = 64
    model.dropout = 0.1
    model.nonlin = "elu"
    model.enable_noise_model = True
    model.analytical_kl = True
    
    loss = config.loss
    
    training = config.training
    training.lr = 1e-3
    training.lr_scheduler_patience = 15 
    training.batch_size = 32
    training.grad_clip_norm_value = 0.5  # Taken from https://github.com/openai/vdvae/blob/main/hps.py#L38
    training.gradient_clip_algorithm = 'value'
    training.earlystop_patience = 100
    training.precision = 32
    training.pre_trained_ckpt_fpath = ''

    return config
