# @mock.patch("careamics.utils.wandb.wandb")
# def test_wandb_logger(wandb, tmp_path: Path, minimum_config: dict):
#     config = Configuration(**minimum_config)
#     logger = WandBLogging(
#         experiment_name="test", log_path=tmp_path, config=config, model_to_watch=None
#     )

#     logger.log_metrics({"acc": 0.0})
#     wandb.init().log.assert_called_once_with({"acc": 0.0}, commit=True)


# @mock.patch("careamics.utils.wandb.wandb")
# def test_wandb_logger_engine(wandb, minimum_config: dict):
#     config = Configuration(**minimum_config)
#     config.training.use_wandb = True
#     engine = Engine(config=config)
#     if engine.use_wandb:
#         assert engine.wandb is not None
#         assert wandb.run
