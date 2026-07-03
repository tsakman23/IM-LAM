import random
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Dict

import rich
from lightning.fabric import Fabric
from lightning.fabric.utilities.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf
from rich.syntax import Syntax

# Suppress wandb Lightning Fabric integration warnings by patching termwarn
import wandb

original_termwarn = wandb.termwarn


def patched_termwarn(message, **kwargs):
    """Suppress specific wandb Lightning Fabric integration warnings."""
    if "This integration is tested and supported for lightning Fabric" not in message:
        original_termwarn(message, **kwargs)


wandb.termwarn = patched_termwarn

from wandb.integration.lightning.fabric import WandbLogger

from ifo.common.utils.utility import set_cuda_backend


class Experiment(ABC):
    """Abstract base class for experiments."""

    stages = ["stage_1"]
    config = {"stage_1": ""}

    @rank_zero_only
    def _print_config(self, cfg: DictConfig) -> None:
        """Print the experiment configuration.

        Args:
            cfg (DictConfig): Experiment configuration.
        """
        OmegaConf.resolve(cfg)
        yaml_str = OmegaConf.to_yaml(cfg)
        syntax = Syntax(yaml_str, "yaml", theme="native", line_numbers=False)
        rich.print(syntax)

    @rank_zero_only
    def _configure_wandb_logging(self, fabric: Fabric, cfg: DictConfig) -> None:
        """Configure Weights & Biases logging for fabric.

        Args:
            fabric (Fabric): Fabric instance.
            cfg (DictConfig): Experiment configuration.
        """
        config = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(config, dict)
        logger_conf = config["logger"]
        # Honor an explicitly-configured logger.name as the W&B display name;
        # otherwise fall back to the stage run id. The run identity / resume key
        # is always the stage run id, independent of the display name.
        logger_conf["name"] = logger_conf.get("name") or config["stage_run_id"]
        logger_conf["id"] = config["stage_run_id"]
        logger_conf["tags"] += [config["env"]["name"][:64]]  # Tags are limited to 64 characters.
        wandb_logger = WandbLogger(**logger_conf, config=config)
        fabric._loggers = [wandb_logger]

    def _configure_fabric(self, cfg: DictConfig) -> Fabric:
        """Configure Fabric.

        Configures Fabric, sets the CUDA backend, and the random seed.

        Args:
            cfg (DictConfig): Experiment configuration.

        Returns:
            Fabric: Configured Fabric instance.
        """
        # Configure CUDA backend.
        self._configure_cuda_backend()

        # Initialize Fabric.
        fabric = Fabric(**cfg["fabric"])
        fabric.launch()

        # Configure random seed for everything.
        self._configure_seed(cfg, fabric)
        return fabric

    def _configure_seed(self, cfg: DictConfig, fabric: Fabric) -> None:
        """Configure random seed.

        If not explicitly set, a random seed is generated, which is used to seed the pseudo-random number generators
        in: torch, numpy, and Python’s random module.
        Args:
            cfg (DictConfig): Experiment configuration.
            fabric (Fabric): Fabric instance.
        """
        if cfg.trainer.random_seed is None:
            cfg.trainer.random_seed = random.randint(0, 1000000)
        fabric.seed_everything(cfg.trainer.random_seed)

    def _configure_cuda_backend(self) -> None:
        """Configure CUDA backend."""
        set_cuda_backend()

    def filter_state_dict_by_prefix(self, state_dict: Dict[str, Any], prefix: str) -> OrderedDict:
        """
        Filters the input state_dict to only include keys that start with the given prefix.
        The prefix is removed from the keys in the resulting OrderedDict.

        Args:
            state_dict (Dict[str, Any]): The state dictionary to filter.
            prefix (str): The prefix to filter by (e.g., 'net.encoder').

        Returns:
            OrderedDict: A filtered and re-keyed OrderedDict.

        Example:
            >>> state_dict = {
            ...     "net.encoder.conv1.weight": "w1",
            ...     "net.encoder.conv1.bias": "b1",
            ...     "net.decoder.fc.weight": "w2"
            ... }
            >>> filter_state_dict_by_prefix(state_dict, "net.encoder")
            OrderedDict([
                ('conv1.weight', 'w1'),
                ('conv1.bias', 'b1')
            ])
        """
        if not prefix.endswith("."):
            prefix += "."

        filtered_dict = OrderedDict()
        for key, value in state_dict.items():
            if key.startswith(prefix):
                new_key = key[len(prefix) :]
                filtered_dict[new_key] = value

        return filtered_dict

    @abstractmethod
    def stage_1(self, cfg: DictConfig) -> None:
        """Start training of stage 1."""
        pass
