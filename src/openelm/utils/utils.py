from dataclasses import is_dataclass
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def validate_config(config):
    if isinstance(config, (str, Path)):
        config = OmegaConf.load(config)
        try:
            return OmegaConf.to_object(config)
        except ValueError:
            return config
    elif isinstance(config, (dict, DictConfig)):
        return DictConfig(config)
    elif is_dataclass(config):
        return config
    else:
        try:
            return OmegaConf.load(config)
        except IOError:
            raise IOError(
                "Invalid config type. Must be a path to a yaml, a dict, or dataclass."
            )
