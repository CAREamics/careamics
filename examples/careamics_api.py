from careamics import CAREamist, Configuration


def main():
    config_dict = {
        "experiment_name": "ConfigTest",
        "working_directory": ".",
        "algorithm": {
            "algorithm_type": "n2v",
            "loss": "n2v",
            "model": {
                "architecture": "UNet",
                "is_3D": False,
            },
            "optimizer": {
                "name": "Adam",
            },
            "lr_scheduler": {"name": "ReduceLROnPlateau"},
        },
        "training": {
            "num_epochs": 1,
            "batch_size": 2,
        },
        "data": {
            "in_memory": True,
            "data_format": "tif",
            "patch_size": [64, 64],
            "axes": "YX",
        },
    }
    config = Configuration(**config_dict)

    careamist = CAREamist(configuration=config)

    train_data = "examples/2D/data/n2v_sem/train/"
    val_data = "examples/2D/data/n2v_sem/val/"

    careamist.train_on_path(train_data, val_data)

    # pred = careamist.predict_on_path(val_data)
    # print(pred.shape)


if __name__ == "__main__":
    main()
