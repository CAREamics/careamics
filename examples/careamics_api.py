from careamics import CAREamist, Configuration

config_dict ={
    "experiment_name": "ConfigTest",
    "working_directory": ".",
    "algorithm": {
        "algorithm_type": "n2v",
        "model": {
           "architecture": "UNet",
            "is_3D": False,
        },
        "loss": "n2v",
        "transforms": {
            "ManipulateN2V": ""
        }
    },
    "training": {
        "num_epochs": 42,
        "batch_size": 16,
        "patch_size": [64, 64],
        "optimizer": {
            "name": "Adam",
        },
        "lr_scheduler": {
            "name": "ReduceLROnPlateau"
        },
        "augmentation": True,
    },
    "data": {
        "in_memory": True,
        "data_format": "tif",
        "axes": "YX",
    },
}
config = Configuration(**config_dict)

careamist = CAREamist(configuration=config)

train_data = "examples/2D/data/n2v_sem/train/"
val_data = "examples/2D/data/n2v_sem/val/"

careamist.train_on_path(train_data, val_data)

pred = careamist.predict_on_path(val_data)

print(pred.shape)