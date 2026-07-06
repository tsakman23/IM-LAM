"""In-process, single-run SLAPO / MaskLAM pipeline runner.

Runs the full pipeline - stage_1 (IDM) -> stage_2 (latent policy) -> stage_3 (BC) and a
final rollout eval - in ONE process and ONE W&B run. Each stage logs under a ``{stage}/``
prefix with its own 0-based step axis (via ``wandb.define_metric``); the final eval logs
under ``eval/``. Per-stage checkpoints (``${run_id}-N``) and ``--selected_stages`` are
preserved. The former subprocess-per-stage mechanism has been removed.

CLI (unchanged from before): global ``key=value`` overrides, plus per-stage
``--stage <name> [-cn <config>] key=value ...`` and ``--selected_stages=[...]``.
Add ``eval=false`` to skip the final rollout eval.

Base Paper (LAPO): https://arxiv.org/abs/2312.10812
"""
import os
import sys
from typing import List, Optional, Tuple

import coolname
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from lightning.fabric import Fabric
from omegaconf import DictConfig, OmegaConf
from wandb.integration.lightning.fabric import WandbLogger

from ifo.common.runner import Runner
from ifo.common.utils.utility import add_legacy_features_to_vector_wrapper, display_banner
from ifo.modules.slapo.experiment import SLAPOExperiment

# Import the reusable in-process rollout eval (repo root must be on sys.path).
sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "submission2026"))
from eval_slapo import success_eval  # noqa: E402

CONFIG_DIR = os.path.join(os.getcwd(), "experiments", "configs")
CONFIG_MAP = SLAPOExperiment.config  # {stage_name: default_config_name}


def _get_override(args: List[str], key: str, default: Optional[str] = None) -> Optional[str]:
    """Return the value of a ``key=value`` override from an arg list (last wins)."""
    prefix = f"{key}="
    value = default
    for arg in args:
        if arg.startswith(prefix):
            value = arg.split("=", 1)[1]
    return value


def _split_config_name(args: List[str]) -> Tuple[Optional[str], List[str]]:
    """Split ``-cn``/``--config-name`` out of an arg list; return (config_name, overrides)."""
    config_name: Optional[str] = None
    overrides: List[str] = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("-cn", "--config-name"):
            config_name = args[i + 1]
            i += 2
            continue
        if arg.startswith("--config-name="):
            config_name = arg.split("=", 1)[1]
            i += 1
            continue
        overrides.append(arg)
        i += 1
    return config_name, overrides


def _compose_stage(
    stage: str,
    global_overrides: List[str],
    stage_args: dict,
    run_id: str,
    checkpoint_dir: str,
    extra: Optional[List[str]] = None,
) -> DictConfig:
    """Compose a single stage's Hydra config in-process (mirrors the old per-subprocess compose)."""
    stage_config_name, stage_overrides = _split_config_name(stage_args.get(stage, []))
    config_name = stage_config_name or CONFIG_MAP[stage]
    overrides = (
        list(global_overrides)
        + list(stage_overrides)
        + [
            f"run_id={run_id}",
            f"trainer.checkpoint_dir={checkpoint_dir}",
            f"trainer.log_prefix={stage}",
        ]
        + list(extra or [])
    )
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name=config_name, overrides=overrides)


def main() -> None:
    run_id_default = coolname.generate_slug(3)
    default_global = [f"run_id={run_id_default}", "trainer.checkpoint_dir=./checkpoints"]

    runner = Runner(SLAPOExperiment)  # reuse the battle-tested arg parser
    global_args, stage_args, selected = runner.parse_arguments(sys.argv[1:])
    global_args, stage_args, selected = runner.merge_arguments(default_global, None, global_args, stage_args, selected)
    _, global_overrides = _split_config_name(global_args)

    run_id = _get_override(global_args, "run_id", run_id_default)
    checkpoint_dir = _get_override(global_args, "trainer.checkpoint_dir", "./checkpoints")
    run_eval = str(_get_override(global_args, "eval", "true")).lower() != "false"
    # `eval` is an orchestrator-level flag, not a Hydra config key - keep it out of compose overrides.
    global_overrides = [o for o in global_overrides if not o.startswith("eval=")]

    display_banner()
    print(f"[pipeline] run_id={run_id} | stages={selected} | eval={run_eval}")

    # One Fabric + one W&B run for the whole pipeline, from the first selected stage's config.
    base_cfg = _compose_stage(selected[0], global_overrides, stage_args, run_id, checkpoint_dir)
    logger_conf = OmegaConf.to_container(base_cfg.logger, resolve=True)
    logger_conf["name"] = run_id
    logger_conf["id"] = run_id
    logger_conf["tags"] = list(logger_conf.get("tags") or []) + [str(base_cfg.env.name)[:64]]
    wandb_logger = WandbLogger(**logger_conf, config=OmegaConf.to_container(base_cfg, resolve=True))
    fabric = Fabric(**OmegaConf.to_container(base_cfg.fabric, resolve=True), loggers=[wandb_logger])
    fabric.launch()
    run = wandb_logger.experiment

    add_legacy_features_to_vector_wrapper()
    experiment = SLAPOExperiment()
    experiment._shared_fabric = fabric
    experiment._shared_run = run

    for stage in selected:
        cfg = base_cfg if stage == selected[0] else _compose_stage(
            stage, global_overrides, stage_args, run_id, checkpoint_dir
        )
        run.define_metric(f"{stage}/step")
        run.define_metric(f"{stage}/*", step_metric=f"{stage}/step")
        print(f"[pipeline] ===== {stage} =====")
        getattr(experiment, stage)(cfg)

    if run_eval and "stage_3" in selected:
        checkpoint = f"{checkpoint_dir}/{run_id}-3"
        eval_cfg = _compose_stage(
            "stage_3", global_overrides, stage_args, run_id, checkpoint_dir,
            extra=[f"trainer.previous_stage_checkpoint={checkpoint}"],
        )
        print(f"[pipeline] ===== eval (checkpoint {checkpoint}) =====")
        metrics = success_eval(eval_cfg, fabric)
        run.log({f"eval/{key}": value for key, value in metrics.items()})
        print(f"[pipeline] eval: {metrics}")

    run.finish()


if __name__ == "__main__":
    main()
