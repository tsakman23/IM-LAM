from typing import Callable, Optional

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from tensordict import TensorDict
from torch import Tensor, nn

import wandb
from ifo.common.module import SupervisedLightningModule
from ifo.common.utils.actions import get_action_metrics
from ifo.common.utils.render import make_comp_grid
from ifo.modules.ppo.module import PPOModule


class LAPOIDMModule(SupervisedLightningModule):
    """Module for learning the inverse dynamics model in LAPO."""

    def __init__(
        self,
        net: nn.Module,
        batch_size: int,
        optimizer: DictConfig,
        lr_scheduler: Optional[DictConfig] = None,
        num_workers: int = 0,
        max_grad_norm: Optional[float] = None,
        debug_transform: Optional[Callable] = None,
        reconstruction_loss_scale_factor: float = 1.0,
        vq_loss_scale_factor: float = 1.0,
        action_loss_scale_factor: float = 1.0,
        **kwargs,
    ) -> None:
        """Instantiate module for learning an inverse dynamics model.

        Args:
            net (nn.Module): Policy neural network.
            batch_size (int): Batch size.
            optimizer (DictConfig): The optimizer to use.
            lr_scheduler (Optional[DictConfig]): The learning rate scheduler to use.
            num_workers (int, optional): Number of workers for dataloader.
            max_grad_norm (Optional[float], optional): Maximum gradient norm. If not provided, no clipping will be done.
            debug_transform (Optional[Callable], optional): Transform to convert observations back to a human
                interpretable format.
            vq_loss_scale_factor (float, optional): Scale factor for the VQ loss. Defaults to 1.0.
            reconstruction_loss_scale_factor (float, optional): Scale factor for the reconstruction loss. Defaults to 1.0.
            action_loss_scale_factor (float, optional): Scale factor for the action loss. Defaults to 1.0.
        """
        super().__init__(net, batch_size, optimizer, lr_scheduler, num_workers, max_grad_norm, **kwargs)
        self.debug_transform = debug_transform
        self.vq_loss_scale_factor = vq_loss_scale_factor
        self.reconstruction_loss_scale_factor = reconstruction_loss_scale_factor
        self.action_loss_scale_factor = action_loss_scale_factor

    @torch.no_grad()
    @torch.compiler.disable()
    def _log_debug_images(self, predicted_next_observation: Tensor, batch: TensorDict, log_prefix: str) -> None:
        assert self.debug_transform is not None, "Debug transform is not provided"
        pred_next_obs = self.debug_transform(predicted_next_observation)
        gt_next_obs = self.debug_transform(batch["observation"][:, -1])
        self.logger.experiment.log(
            {
                f"{log_prefix}/pred_next_obs": wandb.Image(
                    make_comp_grid(pred_next_obs, gt_next_obs),
                    caption="Predicted next state vs ground truth",
                )
            },
        )

    def _forward(self, batch: TensorDict, batch_idx: int, batch_len: int, prefix: str) -> TensorDict:
        """Forward pass.

        Args:
            batch (TensorDict): Batch of observations.
            batch_idx (int): Index of batch.
            batch_len (int): Length of batch.
            prefix (str): Prefix for logging.

        Returns:
            TensorDict: Dictionary with loss and other metrics to log.
        """
        next_observation, action_distribution, vq_loss, perplexity = self.net(batch["observation"])

        reconstruction_loss = F.mse_loss(next_observation, batch["observation"][:, -1])
        action_loss = -action_distribution.log_prob(batch["action"][:, -2]).mean()
        loss = (
            self.reconstruction_loss_scale_factor * reconstruction_loss +
            self.vq_loss_scale_factor * vq_loss +
            self.action_loss_scale_factor * action_loss
        )

        step_dict = TensorDict({
            "loss": loss,
            "reconstruction_loss": reconstruction_loss,
            "action_loss": action_loss,
            "vq_loss": vq_loss,
            "perplexity": perplexity
            })
        step_dict = get_action_metrics(
            action_distribution,
            batch["action"][:, -2],
            prefix="action_decoder_",
            metrics=step_dict
        )

        # Log predicted next observation vs ground truth on the last batch.
        if batch_idx == batch_len - 1 and self.debug_transform is not None:
            self._log_debug_images(next_observation, batch, prefix)

        return step_dict

    def training_step(self, batch: TensorDict, batch_idx: int, batch_len: int) -> TensorDict:
        return self._forward(batch, batch_idx, batch_len, "train")

    def validation_step(self, batch: TensorDict, batch_idx: int, batch_len: int) -> TensorDict:
        return self._forward(batch, batch_idx, batch_len, "val")

    def predict_step(self, batch: TensorDict, batch_idx: int, batch_len: int) -> TensorDict:
        """Predict latent action with IDM.

        Args:
            batch (TensorDict): Batch of observations.
            batch_idx (int): Index of batch.
            batch_len (int): Length of batch.

        Returns:
            TensorDict: Dictionary with latent actions.
        """
        latent_action = self.net.label(batch["observation"])
        return TensorDict({"latent_action": latent_action})


class LAPOLatentActionDecoderModule(SupervisedLightningModule):
    """Module for learning the latent action decoder in LAPO."""

    def training_step(self, batch: TensorDict, batch_idx: int, batch_len: int) -> TensorDict:
        action_distribution = self.net(batch["latent_action"])
        # Take loss with respect to a_t and not a_{t+1}.
        loss = -action_distribution.log_prob(batch["action"][:, -2]).mean()
        return TensorDict({"loss": loss})

    def validation_step(self, batch: TensorDict, batch_idx: int, batch_len: int) -> TensorDict:
        action_distribution = self.net(batch["latent_action"])
        metrics = get_action_metrics(action_distribution, batch["action"][:, -2])
        # Take loss with respect to a_t and not a_{t+1}.
        loss = -action_distribution.log_prob(batch["action"][:, -2]).mean()
        metrics["loss"] = loss
        return TensorDict(metrics)


class LAPOLatentPolicyModule(SupervisedLightningModule):
    """Module for learning the latent policy model in LAPO."""
    def __init__(
        self,
        net: nn.Module,
        batch_size: int,
        optimizer: DictConfig,
        lr_scheduler: Optional[DictConfig] = None,
        num_workers: int = 0,
        max_grad_norm: Optional[float] = None,
        latent_action_loss_scale_factor: float = 1.0,
        action_loss_scale_factor: float = 1.0,
        **kwargs,
    ) -> None:
        """Instantiate module for learning a latent policy.

        Args:
            net (nn.Module): Policy neural network.
            batch_size (int): Batch size.
            optimizer (DictConfig): The optimizer to use.
            lr_scheduler (Optional[DictConfig]): The learning rate scheduler to use.
            num_workers (int, optional): Number of workers for dataloader.
            max_grad_norm (Optional[float], optional): Maximum gradient norm. If not provided, no clipping will be done.
            latent_action_loss_scale_factor (float, optional): Scale factor for the latent action loss. Defaults to 1.0.
            action_loss_scale_factor (float, optional): Scale factor for the action loss. Defaults to 1.0.
        """
        super().__init__(net, batch_size, optimizer, lr_scheduler, num_workers, max_grad_norm, **kwargs)
        self.latent_action_loss_scale_factor = latent_action_loss_scale_factor
        self.action_loss_scale_factor = action_loss_scale_factor

    def training_step(self, batch: TensorDict, batch_idx: int, batch_len: int) -> TensorDict:
        latent_action, predicted_latent_action, action_distribution = self.net(batch["observation"])
        latent_action_loss = F.mse_loss(predicted_latent_action, latent_action)
        action_loss = -action_distribution.log_prob(batch["action"][:, -2]).mean()
        loss = (
            self.latent_action_loss_scale_factor * latent_action_loss +
            self.action_loss_scale_factor * action_loss
        )
        step_dict = TensorDict({
            "loss": loss,
            "latent_action_loss": latent_action_loss,
            "action_loss": action_loss
        })
        step_dict = get_action_metrics(
            action_distribution,
            batch["action"][:, -2],
            prefix="action_decoder_",
            metrics=step_dict
            )
        return step_dict

    def validation_step(self, batch: TensorDict, batch_idx: int, batch_len: int) -> TensorDict:
        return self.training_step(batch, batch_idx, batch_len)

    def predict_step(self, batch: TensorDict, batch_idx: int, batch_len: int) -> TensorDict:
        """Predict latent action with latent policy.

        Args:
            batch (TensorDict): Batch of observations.
            batch_idx (int): Index of batch.
            batch_len (int): Length of batch.

        Returns:
            TensorDict: Dictionary with latent actions.
        """
        latent_action = self.net.label(batch["observation"])
        return TensorDict({"latent_action": latent_action})


class LAPOPPOModule(PPOModule):
    """PPO module for LAPO."""

    def predict_step(self, batch: TensorDict, batch_idx: int, batch_len: int) -> TensorDict:
        """Training step for the action decoder.

        We are just calling it predict_step to avoid problems with fabric. Should be called training_step_decoder.

        Args:
            batch (TensorDict): Batch of latent actions and true actions.
            batch_idx (int): Index of batch.
            batch_len (int): Length of batch.

        Returns:
            TensorDict: Dictionary with loss.
        """
        with torch.no_grad():
            latent_action = self.net.label(batch["observation"])
        action_distribution = self.net.decoder(latent_action)
        loss = -action_distribution.log_prob(batch["action"][:, -2]).mean()
        return TensorDict({"loss": loss})
