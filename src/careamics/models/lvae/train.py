from .data_utils import (
    DataSplitType
)
from .data_modules import (
    LCMultiChDloader,
    MultiChDloader
)

def create_dataset(
    config,
    datadir,
    eval_datasplit_type=DataSplitType.Val,
    raw_data_dict=None,
    skip_train_dataset=False,
    kwargs_dict=None
):

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
    if 'multiscale_lowres_count' in config.data and config.data.multiscale_lowres_count is not None:
        # Get padding attributes
        if 'padding_kwargs' not in kwargs_dict:
            padding_kwargs = {}
            if 'padding_mode' in config.data and config.data.padding_mode is not None:
                padding_kwargs['mode'] = config.data.padding_mode
            else:
                padding_kwargs['mode'] = 'reflect'
            if 'padding_value' in config.data and config.data.padding_value is not None:
                padding_kwargs['constant_values'] = config.data.padding_value
            else:
                padding_kwargs['constant_values'] = None
        else:
            padding_kwargs = kwargs_dict.pop('padding_kwargs')


        train_data = None if skip_train_dataset else LCMultiChDloader(
            config.data,
            datapath,
            datasplit_type=DataSplitType.Train,
            val_fraction=0.1,
            test_fraction=0.1,
            normalized_input=normalized_input,
            use_one_mu_std=use_one_mu_std,
            enable_rotation_aug=train_aug_rotate,
            enable_random_cropping=enable_random_cropping,
            num_scales=config.data.multiscale_lowres_count,
            lowres_supervision=lowres_supervision,
            padding_kwargs=padding_kwargs,
            **kwargs_dict,
            allow_generation=True
        )
        max_val = train_data.get_max_val()

        val_data = LCMultiChDloader(
            config.data,
            datapath,
            datasplit_type=eval_datasplit_type,
            val_fraction=0.1,
            test_fraction=0.1,
            normalized_input=normalized_input,
            use_one_mu_std=use_one_mu_std,
            enable_rotation_aug=False,  # No rotation aug on validation
            enable_random_cropping=False,
            # No random cropping on validation. Validation is evaluated on determistic grids
            num_scales=config.data.multiscale_lowres_count,
            lowres_supervision=lowres_supervision,
            padding_kwargs=padding_kwargs,
            allow_generation=False,
            **kwargs_dict,
            max_val=max_val,
        )
    # 2) Vanilla data loader
    else:
        train_data_kwargs = {'allow_generation': True, **kwargs_dict}
        val_data_kwargs = {'allow_generation': False, **kwargs_dict}
        
        train_data_kwargs['enable_random_cropping'] = enable_random_cropping
        val_data_kwargs['enable_random_cropping'] = False

        train_data = None if skip_train_dataset else MultiChDloader(
            config.data,
            datapath,
            datasplit_type=DataSplitType.Train,
            val_fraction=0.1,
            test_fraction=0.1,
            normalized_input=normalized_input,
            use_one_mu_std=use_one_mu_std,
            enable_rotation_aug=train_aug_rotate,
            **train_data_kwargs
        )

        max_val = train_data.get_max_val()
        val_data = MultiChDloader(
            config.data,
            datapath,
            datasplit_type=eval_datasplit_type,
            val_fraction=0.1,
            test_fraction=0.1,
            normalized_input=normalized_input,
            use_one_mu_std=use_one_mu_std,
            enable_rotation_aug=False,  # No rotation aug on validation
            max_val=max_val,
            **val_data_kwargs,
        )

    # For normalizing, we should be using the training data's mean and std.
    mean_val, std_val = train_data.compute_mean_std()
    train_data.set_mean_std(mean_val, std_val)
    val_data.set_mean_std(mean_val, std_val)

    return train_data, val_data
