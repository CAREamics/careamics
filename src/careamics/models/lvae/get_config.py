"""
Here there are functions to define a config file.
"""
import os
import ml_collections

from careamics.models.lvae.data_utils import DataType

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
    data.num_channels = 2
    
    model = config.model
    model.z_dims = [128, 128, 128, 128]
    model.n_filters = 64
    model.dropout = 0.1
    model.nonlin = "elu"
    model.enable_noise_model = True
    model.analytical_kl = False
    
    loss = config.loss
    
    training = config.training
    training.lr = 1e-3
    training.lr_scheduler_patience = 15
    training.batch_size = 32
    training.earlystop_patience = 100
    training.max_epochs = 400
    training.pre_trained_ckpt_fpath = ''
    
    # Set of attributes not to include in the PyDantic data model
    training.num_workers = 4 
    training.grad_clip_norm_value = 0.5  # Taken from https://github.com/openai/vdvae/blob/main/hps.py#L38
    training.gradient_clip_algorithm = 'value'
    training.precision = 32
    data.data_type = DataType.BioSR_MRC
    data.ch1_fname = 'ER/GT_all.mrc'
    data.ch2_fname = 'Microtubules/GT_all.mrc'
    fname = '/group/jug/federico/careamics_training/noise_models/139/GMMNoiseModel_BioSR-__6_4_Clip0.0-1.0_Sig0.125_UpNone_Norm0_bootstrap.npz'
    model.noise_model_ch1_fpath = fname
    model.noise_model_ch2_fpath = fname
    
    return config
