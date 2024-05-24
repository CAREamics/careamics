from omegaconf import DictConfig, OmegaConf
import hydra

from .configuration import Config

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.missing_keys(cfg))
    print(OmegaConf.resolve(cfg))
    print(OmegaConf.to_yaml(cfg))