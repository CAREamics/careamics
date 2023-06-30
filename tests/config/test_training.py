from careamics_restoration.config.training import Optimizer


# TODO this is not a good test replace or remove
def test_optimizer_enum(minimum_config):
    optim_dict = minimum_config["training"]["optimizer"]

    opt = Optimizer(**optim_dict)

    assert opt.dict()["name"] == optim_dict["name"]
