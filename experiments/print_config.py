"""
Print the config for a given experiment.

Usage:
python experiments/print_config.py -cn bc_default
"""

import hydra
from omegaconf import OmegaConf


@hydra.main(version_base=None, config_path="configs")
def print_config(cfg):
    # Convert config to dict and print it
    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    print_config()
