from pytorch_lightning import LightningModule

from careamics import CAREamicsModule

def test_careamics_module(minimum_algorithm):
    """Test CAREamicsModule class as an intermediate layer."""
    # extract model parameters
    model_parameters = minimum_algorithm["model"].copy()
    model_parameters.pop("architecture")

    # extract optimizer and scheduler parameters
    opt = minimum_algorithm["optimizer"]
    optimizer_parameters = opt["parameters"] if "parameters" in opt else None

    lr = minimum_algorithm["lr_scheduler"]
    lr_scheduler_parameters = lr["parameters"] if "parameters" in lr else None

    # instantiate CAREamicsModule
    module = CAREamicsModule(
        algorithm_type=minimum_algorithm["algorithm_type"],
        loss=minimum_algorithm["loss"],
        architecture=minimum_algorithm["model"]["architecture"],
        model_parameters=model_parameters,
        optimizer=opt["name"],
        optimizer_parameters=optimizer_parameters,
        lr_scheduler=lr["name"],
        lr_scheduler_parameters=lr_scheduler_parameters,
    )

    assert isinstance(module, LightningModule)