from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, cast

import hydra
import torch
from calculate_success_rate import calculate_success_rate
from lightning.fabric import Fabric
from omegaconf import DictConfig, OmegaConf

import wandb
from ifo.common.collector import Collector, ExplorationMode
from ifo.common.utils.utility import (
    add_legacy_features_to_vector_wrapper,
    get_latest_checkpoint,
    set_cuda_backend,
)


def load_checkpoint_state_dict(checkpoint_path: str) -> Dict[str, Any]:
    resolved = get_latest_checkpoint(checkpoint_path)
    if resolved is None:
        raise FileNotFoundError(f"Could not resolve checkpoint from path: {checkpoint_path}")
    payload = torch.load(resolved, map_location="cpu")
    if isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], dict):
        return payload["model"]
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"Unexpected checkpoint format in {resolved}")


def filter_state_dict_by_prefix(state_dict: Dict[str, Any], prefix: str) -> OrderedDict[str, Any]:
    if not prefix.endswith("."):
        prefix += "."
    filtered_dict: OrderedDict[str, Any] = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith(prefix):
            filtered_dict[key[len(prefix) :]] = value
    return filtered_dict


def maybe_log_to_wandb(cfg: DictConfig, metrics: Dict[str, float]) -> None:
    if str(cfg.logger.mode) != "online":
        return
    config = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
    logger_conf = dict(config["logger"])
    run_name = str(config.get("stage_run_id") or config.get("run_id"))
    logger_conf["name"] = run_name
    logger_conf["id"] = run_name + "___"
    tags = list(logger_conf.get("tags") or [])
    tags.append(str(config["env"]["name"])[:64])
    logger_conf["tags"] = tags
    run = wandb.init(**logger_conf, config=config)
    if run is not None:
        run.log(metrics)
        run.finish()


def resolve_required_checkpoint(cfg: DictConfig) -> str:
    value = OmegaConf.select(cfg, "trainer.previous_stage_checkpoint")
    if value is None:
        raise ValueError(
            "Missing `trainer.previous_stage_checkpoint`. Pass it as a Hydra override if needed."
        )
    return str(value)


def configure_fabric(cfg: DictConfig) -> Fabric:
    set_cuda_backend()
    fabric = Fabric(**cfg["fabric"])
    fabric.launch()
    if cfg.trainer.random_seed is not None:
        fabric.seed_everything(cfg.trainer.random_seed)
    return fabric


@hydra.main(version_base=None, config_path="../../experiments/configs", config_name="lapo_bc_default_stage_3")
def main(cfg: DictConfig) -> None:
    add_legacy_features_to_vector_wrapper()
    fabric = configure_fabric(cfg)

    checkpoint_path = resolve_required_checkpoint(cfg)

    env = hydra.utils.instantiate(cfg.env, num_envs=cfg.env.num_envs)
    try:
        net = hydra.utils.instantiate(cfg.net)(
            observation_space=env.single_observation_space,
            action_space=env.single_action_space,
        )
        module = hydra.utils.instantiate(cfg.module, net=net)
        state_dict = load_checkpoint_state_dict(checkpoint_path)
        net.load_state_dict(filter_state_dict_by_prefix(state_dict, "net"), strict=False)
        module.training_step = torch.compile(
            module.training_step, mode="max-autotune", disable=not cfg.trainer.compile
        )
        module.validation_step = torch.compile(
            module.validation_step, mode="max-autotune", disable=not cfg.trainer.compile
        )
        module.forward = torch.compile(module.forward, mode="max-autotune", disable=not cfg.trainer.compile)
        module = fabric.setup_module(module)
        module.eval()

        collector = Collector(env=env, policy=module, random_seed=cfg.trainer.random_seed)
        max_steps = int(cfg.rollout_callback.rollout_steps)
        exploration_mode = ExplorationMode(str(cfg.rollout_callback.rollout_exploration_mode))
        success_rate, episode_return, episode_length = calculate_success_rate(
            collector=collector,
            max_steps=max_steps,
            exploration_mode=exploration_mode,
            reset_before_rollout=True,
        )

        metrics = {
            "val/episode_success_rate": float(success_rate),
            "val/episode_return": episode_return,
            "val/episode_length": episode_length,
        }
        print(f"run_id={cfg.run_id}")
        print(f"env={cfg.env.name}")
        print(f"previous_stage_checkpoint={checkpoint_path}")
        print(f"success_rate={success_rate:.6f}")
        print(f"episode_return={episode_return:.6f}")
        print(f"episode_length={episode_length:.6f}")
        maybe_log_to_wandb(cfg, metrics)
    finally:
        env.close()


if __name__ == "__main__":
    main()
