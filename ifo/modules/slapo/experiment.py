import hydra
import torch
from omegaconf import DictConfig

from ifo.common.experiment import Experiment
from ifo.common.transforms.debug import get_debug_transform
from ifo.common.utils.expert_constants import get_action_variance
from ifo.common.utils.utility import (
    Conditional,
    get_latest_checkpoint,
    requires_grad,
)


class SLAPOExperiment(Experiment):
    """Class for SLAPO experiment.

    Note: This experiment does not execute stage_0, which is used to train a UNet-based binary segmentation model on
          ground-truth masks.
    """

    stages = ["stage_1", "stage_2", "stage_3"]
    config = {
        "stage_1": "slapo_default_stage_1",
        "stage_2": "slapo_default_stage_2",
        "stage_3": "slapo_default_stage_3",
    }

    # Shared-Fabric/shared-W&B-run injection (experiments/run_slapo.py) and per-stage metric
    # prefixing / 0-based step axes (each stage's trainer.log_prefix override) are handled by
    # the base Experiment class and each stage's trainer config respectively.

    def stage_0(self, cfg: DictConfig) -> None:
        """Train UNet-based binary segmentation model on ground-truth masks."""
        # Initialize Fabric
        fabric = self._configure_fabric(cfg)

        # Print config.
        self._print_config(cfg)

        # Configure logging
        self._configure_wandb_logging(fabric, cfg)

        # Configure environment.
        env = hydra.utils.instantiate(cfg.env, num_envs=1)
        observation_space = env.single_observation_space
        del env

        # Initialize model.
        net = hydra.utils.instantiate(cfg.net)(observation_space=observation_space)
        module = hydra.utils.instantiate(cfg.module, net=net)

        # Initialize datasets.
        train_dataset = hydra.utils.instantiate(cfg.dataset, split="train")
        val_dataset = hydra.utils.instantiate(cfg.dataset, split="test")

        # Initialize trainer and start training.
        checkpoint_dir = f"{cfg.trainer.checkpoint_dir}/{cfg.stage_run_id}"
        trainer = hydra.utils.instantiate(cfg.trainer, fabric=fabric, checkpoint_dir=checkpoint_dir)
        trainer.fit(
            model=module, train_dataset=train_dataset, val_dataset=val_dataset, checkpoint=cfg.trainer.checkpoint_path
        )

    def stage_1(self, cfg: DictConfig) -> None:
        """Start training of inverse dynamics model.

        Args:
            cfg (DictConfig): Configuration for the experiment.
        """
        # Initialize Fabric
        fabric = self._configure_fabric(cfg)

        # Print config.
        self._print_config(cfg)

        # Configure logging
        self._configure_wandb_logging(fabric, cfg)

        # Configure environment.
        env = hydra.utils.instantiate(cfg.env, num_envs=1)
        action_space = env.single_action_space
        observation_space = env.single_observation_space
        del env

        # Initialize model.
        net = hydra.utils.instantiate(cfg.net)(observation_space=observation_space, action_space=action_space)
        # Var(a) for action_decoder_nmse: auto-resolved from the task in cfg.env.name via
        # the ACTION_VARIANCE table (ifo.common.utils.expert_constants), unless a config/CLI
        # override (module.action_variance=...) is already set, which takes precedence.
        action_variance = cfg.module.get("action_variance")
        if action_variance is None:
            action_variance = get_action_variance(cfg.env.name)
        module = hydra.utils.instantiate(
            cfg.module, net=net, debug_transform=get_debug_transform(cfg.dataset.name),
            log_prefix=cfg.trainer.log_prefix, action_variance=action_variance,
        )

        # Load trained segmentation model if available.
        checkpoint_path = get_latest_checkpoint(cfg.trainer.previous_stage_checkpoint)
        if net.segmentation_net is not None and checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path)["model"]
            net.segmentation_net.load_state_dict(self.filter_state_dict_by_prefix(state_dict, "net"))
            requires_grad(net.segmentation_net, False)

        # Initialize datasets.
        train_dataset = hydra.utils.instantiate(cfg.dataset, split="train")
        val_dataset = hydra.utils.instantiate(cfg.dataset, split="test")

        # Initialize trainer and start training.
        checkpoint_dir = f"{cfg.trainer.checkpoint_dir}/{cfg.stage_run_id}"
        trainer = hydra.utils.instantiate(cfg.trainer, fabric=fabric, checkpoint_dir=checkpoint_dir)
        # Optional extraction-beta annealing curriculum (absent by default -> no-op).
        beta_anneal_cfg = cfg.get("extraction_beta_anneal", None)
        if beta_anneal_cfg is not None:
            fabric._callbacks.append(hydra.utils.instantiate(beta_anneal_cfg))
        trainer.fit(
            model=module,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            checkpoint=cfg.trainer.checkpoint_path,
            load_only_model=cfg.trainer.load_only_model,
        )

    def stage_2(self, cfg: DictConfig) -> None:
        """Train latent policy.

        Args:
            cfg (DictConfig): Configuration for the experiment.
        """
        # Initialize Fabric
        fabric = self._configure_fabric(cfg)

        # Print config.
        self._print_config(cfg)

        # Configure logging
        self._configure_wandb_logging(fabric, cfg)

        # Configure environment.
        env = hydra.utils.instantiate(cfg.env, num_envs=1)
        action_space = env.single_action_space
        observation_space = env.single_observation_space
        del env

        # Initialize model and load trained inverse dynamics model.
        net = hydra.utils.instantiate(cfg.net)(observation_space=observation_space, action_space=action_space)
        checkpoint_path = get_latest_checkpoint(cfg.trainer.previous_stage_checkpoint)
        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path)["model"]
            net.inverse_dynamics_model.load_state_dict(self.filter_state_dict_by_prefix(state_dict, "net.encoder"))
            requires_grad(net.inverse_dynamics_model, False)

            # Load trained segmentation model if available.
            if net.segmentation_net is not None:
                net.segmentation_net.load_state_dict(self.filter_state_dict_by_prefix(state_dict, "net.segmentation_net"))
                requires_grad(net.segmentation_net, False)

        module = hydra.utils.instantiate(cfg.module, net=net)

        # Initialize datasets.
        train_dataset = hydra.utils.instantiate(cfg.dataset, split="train")
        val_dataset = hydra.utils.instantiate(cfg.dataset, split="test")

        # Initialize trainer and start training.
        checkpoint_dir = f"{cfg.trainer.checkpoint_dir}/{cfg.stage_run_id}"
        trainer = hydra.utils.instantiate(cfg.trainer, fabric=fabric, checkpoint_dir=checkpoint_dir)
        trainer.fit(
            model=module, train_dataset=train_dataset, val_dataset=val_dataset, checkpoint=cfg.trainer.checkpoint_path
        )

    def stage_3(self, cfg: DictConfig) -> None:
        """Train policy.

        Args:
            module (LightningModule, optional): Module containing the trained latent policy.
        """
        # Initialize Fabric
        fabric = self._configure_fabric(cfg)

        # Print config.
        self._print_config(cfg)

        # Configure logging
        self._configure_wandb_logging(fabric, cfg)

        # Configure environment.
        # name_prefix scopes the rollout-callback's recorded videos under "stage_3/video" in
        # W&B instead of an unscoped "videos" key shared by any other recorder in the run
        # (see VecVideoRecorder.stop_recording).
        val_env = hydra.utils.instantiate(cfg.env, record_video=True, run_id=cfg.stage_run_id, name_prefix="stage_3")
        action_space = val_env.single_action_space
        observation_space = val_env.single_observation_space

        # Add callbacks.
        rollout_callback = hydra.utils.instantiate(
            cfg.rollout_callback, env=val_env, random_seed=cfg.trainer.random_seed
        )
        fabric._callbacks.append(rollout_callback)

        # Initialize model and load trained inverse dynamics model and latent policy.
        net = hydra.utils.instantiate(cfg.net)(observation_space=observation_space, action_space=action_space)
        checkpoint_path = get_latest_checkpoint(cfg.trainer.previous_stage_checkpoint)
        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path)["model"]
            net.latent_policy.load_state_dict(self.filter_state_dict_by_prefix(state_dict, "net.latent_policy"))
            requires_grad(net.latent_policy, False)
        module = hydra.utils.instantiate(cfg.module, net=net)

        # Initialize datasets.
        train_dataset = hydra.utils.instantiate(cfg.dataset, split="train")
        val_dataset = hydra.utils.instantiate(cfg.dataset, split="test")

        # Take subset of dataset for behavior cloning.
        train_dataset = torch.utils.data.Subset(
            train_dataset, range(min(int(cfg.dataset.subset_size), len(train_dataset))))
        val_dataset = torch.utils.data.Subset(
            val_dataset, range(min(int(cfg.dataset.subset_size), len(val_dataset))))

        # Initialize trainer and start training.
        checkpoint_dir = f"{cfg.trainer.checkpoint_dir}/{cfg.stage_run_id}"
        trainer = hydra.utils.instantiate(cfg.trainer, fabric=fabric, checkpoint_dir=checkpoint_dir)
        trainer.fit(
            model=module, train_dataset=train_dataset, val_dataset=val_dataset, checkpoint=cfg.trainer.checkpoint_path
        )

        # Close environment.
        val_env.close()


# The multi-stage pipeline is driven in-process by experiments/run_slapo.py (one process,
# one W&B run). The former subprocess-per-stage entry point has been removed.
