from dataclasses import dataclass

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf, MISSING

from ..careamist import CAREamist
from ..config import Configuration, configuration_factory

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    # engine = CAREamist(source=Configuration(**cfg))

if __name__ == "__main__":
    main()


