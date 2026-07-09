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
from ifo.modules.slapo.utils import agent_occlusion, dilate_mask, erode_mask


class SLAPOSegmentationModule(SupervisedLightningModule):
    """Module for learning a binary segmentation model in SLAPO stage_0."""

    def __init__(
        self,
        net: nn.Module,
        batch_size: int,
        optimizer: DictConfig,
        lr_scheduler: Optional[DictConfig] = None,
        num_workers: int = 0,
        max_grad_norm: Optional[float] = None,
        **kwargs,
    ) -> None:
        """Instantiate module for learning a segmentation model.

        Args:
            net (nn.Module): Segmentation neural network.
            batch_size (int): Batch size.
            optimizer (DictConfig): The optimizer to use.
            lr_scheduler (Optional[DictConfig]): The learning rate scheduler.
            num_workers (int, optional): Number of workers for dataloader.
            max_grad_norm (Optional[float], optional): Maximum gradient norm.
        """
        super().__init__(net, batch_size, optimizer, lr_scheduler, num_workers, max_grad_norm, **kwargs)

    @torch.no_grad()
    @torch.compiler.disable()
    def _log_debug_masks(self, pred_mask: Tensor, gt_mask: Tensor, log_prefix: str, threshold: float = 0.5) -> None:
        """
        Log predicted vs. ground-truth masks.

        Args:
            pred_mask (Tensor): Predicted mask of shape (B, T, 1, H, W).
            gt_mask (Tensor): Ground truth mask of shape (B, T, 1, H, W).
            log_prefix (str): Prefix for logging.
            threshold (float, optional): Threshold for binarizing the masks. Defaults to 0.5.
        """
        pred_mask = (pred_mask > threshold).float()
        self.logger.experiment.log(
            {
                f"{log_prefix}/pred_mask": wandb.Image(
                    make_comp_grid(pred_mask[:, 0], gt_mask[:, 0]),
                    caption="Predicted mask vs ground truth",
                )
            },
        )

    @torch.no_grad()
    def compute_miou(self, pred_masks: Tensor, gt_masks: Tensor, threshold: float = 0.5) -> Tensor:
        """
        Compute mean intersection over union (mIoU).

        Args:
            pred_masks (Tensor): Predicted masks of shape (B, T, 1, H, W).
            gt_masks (Tensor): Ground truth masks of shape (B, T, 1, H, W).
            threshold (float, optional): Threshold for binarizing the masks. Defaults to 0.5.

        Returns:
            Tensor: Mean intersection over union (mIoU).
        """
        pred_masks = (pred_masks > threshold).float()
        intersection = (pred_masks * gt_masks).sum(dim=[2, 3, 4])
        union = (pred_masks + gt_masks).sum(dim=[2, 3, 4]) - intersection
        miou = intersection / (union + 1e-6)
        return miou.mean()

    def _forward(self, batch: TensorDict, batch_idx: int, batch_len: int, prefix: str) -> TensorDict:
        """Forward pass for segmentation training/validation."""
        pred_masks = self.net(batch["observation"])

        loss = F.binary_cross_entropy(pred_masks, batch["mask"])
        miou = self.compute_miou(pred_masks, batch["mask"])

        step_dict = TensorDict(
            {
                "loss": loss,
                "miou": miou,
            }
        )

        # Visualize masks on the last batch of each epoch.
        if batch_idx == batch_len - 1:
            self._log_debug_masks(pred_masks, batch["mask"], prefix)

        return step_dict

    def training_step(self, batch: TensorDict, batch_idx: int, batch_len: int) -> TensorDict:
        return self._forward(batch, batch_idx, batch_len, "train")

    def validation_step(self, batch: TensorDict, batch_idx: int, batch_len: int) -> TensorDict:
        return self._forward(batch, batch_idx, batch_len, "val")

    def predict_step(self, batch: TensorDict, batch_idx: int, batch_len: int) -> TensorDict:
        """Predict masks for evaluation or visualization."""
        images = batch["observation"]
        pred_masks = self.net.label(images)
        return TensorDict({"pred_mask": pred_masks})


class SLAPOIDMModule(SupervisedLightningModule):
    """Module for learning the inverse dynamics model in SLAPO."""

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
        mask_observation: bool = True,
        mask_loss: bool = False,
        occlude_mask_observation: bool = False,
        occlude_mask_observation_fraction: float = 0.5,
        dilate_mask: bool = False,
        dilate_mask_radius: int = 0,
        erode_mask: bool = False,
        erode_mask_radius: int = 0,
        object_mask_loss: bool = False,
        object_mask_input: bool = False,
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
            mask_observation (bool, optional): Whether to mask the observations or not. Defaults to True.
            mask_loss (bool, optional): Whether to mask the loss or not. Defaults to False.
            occlude_mask_observation (bool, optional): Whether to occlude the mask observation or not. Defaults to False.
            occlude_mask_observation_fraction (float, optional): Fraction of the mask observation to occlude. Defaults to 0.5.
            dilate_mask (bool, optional): Whether to dilate the mask or not. Defaults to False.
            dilate_mask_radius (int, optional): Radius of the dilation. Defaults to 0.
            erode_mask (bool, optional): Whether to erode the mask or not. Defaults to False.
            erode_mask_radius (int, optional): Radius of the erosion. Defaults to 0.
            object_mask_loss (bool, optional): Foreground-MaskLAM. If True, gate the
                reconstruction loss by the union of the agent mask and the object
                mask (requires an ``object_mask`` in the batch). Defaults to False.
            object_mask_input (bool, optional): If True, also feed that agent+object
                union as the model's mask input channel (and for observation masking)
                instead of the agent mask alone. Defaults to False (agent-only input,
                matching MaskLAM; the paper's mask-input ablation helps DCS but
                slightly hurts DMW).
        """
        super().__init__(net, batch_size, optimizer, lr_scheduler, num_workers, max_grad_norm, **kwargs)
        self.debug_transform = debug_transform
        self.vq_loss_scale_factor = vq_loss_scale_factor
        self.reconstruction_loss_scale_factor = reconstruction_loss_scale_factor
        self.action_loss_scale_factor = action_loss_scale_factor
        self.mask_observation = mask_observation
        self.mask_loss = mask_loss
        self.occlude_mask_observation = occlude_mask_observation
        self.occlude_mask_observation_fraction = occlude_mask_observation_fraction
        self.dilate_mask = dilate_mask
        self.dilate_mask_radius = dilate_mask_radius
        self.erode_mask = erode_mask
        self.erode_mask_radius = erode_mask_radius
        # Foreground-MaskLAM: weight the reconstruction loss (and optionally the
        # mask input channel) by the union of the agent mask and the manipulated
        # object mask, instead of the agent mask alone.
        self.object_mask_loss = object_mask_loss
        self.object_mask_input = object_mask_input

    @torch.no_grad()
    @torch.compiler.disable()
    def _log_debug_images(self, predicted_next_observation: Tensor, next_observation: Tensor, log_prefix: str) -> None:
        assert self.debug_transform is not None, "Debug transform is not provided"
        pred_next_obs = self.debug_transform(predicted_next_observation)
        gt_next_obs = self.debug_transform(next_observation)
        self.logger.experiment.log(
            {
                f"{log_prefix}/pred_next_obs": wandb.Image(
                    make_comp_grid(pred_next_obs, gt_next_obs),
                    caption="Predicted next state vs ground truth",
                )
            },
        )

    @torch.no_grad()
    @torch.compiler.disable()
    def _log_debug_masks(self, mask: Tensor, gt_mask: Tensor, log_prefix: str) -> None:
        self.logger.experiment.log(
            {
                f"{log_prefix}/mask": wandb.Image(make_comp_grid(mask * 255.0, gt_mask * 255.0), caption="Modified mask vs ground truth"),
            },
        )

    @torch.no_grad()
    def compute_miou(self, pred_masks: Tensor, gt_masks: Tensor, threshold: float = 0.5) -> Tensor:
        """
        Compute mean intersection over union (mIoU).

        Args:
            pred_masks (Tensor): Predicted masks of shape (B, T, 1, H, W).
            gt_masks (Tensor): Ground truth masks of shape (B, T, 1, H, W).
            threshold (float, optional): Threshold for binarizing the masks. Defaults to 0.5.

        Returns:
            Tensor: Mean intersection over union (mIoU).
        """
        pred_masks = (pred_masks > threshold).float()
        intersection = (pred_masks * gt_masks).sum(dim=[2, 3, 4])
        union = (pred_masks + gt_masks).sum(dim=[2, 3, 4]) - intersection
        miou = intersection / (union + 1e-6)
        return miou.mean()

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
        if self.occlude_mask_observation:
            batch = agent_occlusion(batch, self.occlude_mask_observation_fraction, 0.0)

        if self.dilate_mask:
            original_mask = batch["mask"].clone()
            dilated_mask = dilate_mask(batch["mask"], self.dilate_mask_radius)
            miou = self.compute_miou(dilated_mask, batch["mask"])
            batch["mask"] = dilated_mask

        if self.erode_mask:
            original_mask = batch["mask"].clone()
            eroded_mask = erode_mask(batch["mask"], self.erode_mask_radius)
            miou = self.compute_miou(eroded_mask, batch["mask"])
            batch["mask"] = eroded_mask

        # Replace ground-truth mask with predicted mask if segmentation model is available and needed.
        if self.net.segmentation_net is not None and (self.mask_observation or self.mask_loss):
            self.net.segmentation_net.eval()
            with torch.no_grad():
                batch["mask"] = self.net.segmentation_net.label(batch["observation"])

        # Foreground-MaskLAM: optionally fold the manipulated-object mask into the
        # agent mask (pixel-wise union). `input_mask` feeds the model / observation
        # masking; `loss_mask` gates the reconstruction loss. With both flags off
        # this is identical to MaskLAM (both fall back to the agent mask).
        agent_mask = batch.get("mask", None)
        object_mask = batch.get("object_mask", None)
        has_object = object_mask is not None
        input_mask = agent_mask
        loss_mask = agent_mask
        if agent_mask is not None and has_object:
            foreground_mask = (agent_mask + object_mask).clamp(0.0, 1.0)
            if self.object_mask_input:
                input_mask = foreground_mask
            if self.object_mask_loss:
                loss_mask = foreground_mask

        # Mask the observations if the mask is provided and enabled.
        if self.mask_observation and agent_mask is not None:
            batch["observation"] = batch["observation"] * input_mask

        next_observation, action_distribution, vq_loss, perplexity = self.net(batch["observation"], input_mask)

        if self.mask_loss and agent_mask is not None:
            mask = loss_mask[:, self.net.frame_stack]
            reconstruction_loss = F.mse_loss(
                next_observation, batch["observation"][:, self.net.frame_stack], reduction="none")
            reconstruction_loss = (reconstruction_loss * mask).sum() / mask.sum()
        else:
            reconstruction_loss = F.mse_loss(next_observation, batch["observation"][:, self.net.frame_stack])

        action_loss = -action_distribution.log_prob(batch["action"][:, self.net.frame_stack-1]).mean()
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
            batch["action"][:, self.net.frame_stack-1],
            prefix="action_decoder_",
            metrics=step_dict
        )
        if self.dilate_mask or self.erode_mask:
            step_dict["miou"] = miou

        # Log predicted next observation vs ground truth on the last batch.
        if batch_idx == batch_len - 1 and self.debug_transform is not None:
            self._log_debug_images(next_observation, batch["observation"][:, self.net.frame_stack], prefix)
            if self.dilate_mask or self.erode_mask:
                self._log_debug_masks(batch["mask"][:, self.net.frame_stack], original_mask[:, self.net.frame_stack], prefix)

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
        # Replace ground-truth mask with predicted mask if segmentation model is available and needed.
        if self.net.segmentation_net is not None and self.mask_observation:
            with torch.no_grad():
                self.net.segmentation_net.eval()
                batch["mask"] = self.net.segmentation_net.label(batch["observation"])

        if self.mask_observation and "mask" in batch:
            batch["observation"] = batch["observation"] * batch["mask"]

        latent_action = self.net.label(batch["observation"], batch.get("mask", None))
        return TensorDict({"latent_action": latent_action})


class SLAPOLatentPolicyModule(SupervisedLightningModule):
    """Module for learning the latent policy model in SLAPO."""
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
        latent_action, predicted_latent_action, action_distribution = self.net(batch["observation"], batch.get("mask", None))
        latent_action_loss = F.mse_loss(predicted_latent_action, latent_action)
        action_loss = -action_distribution.log_prob(batch["action"][:, self.net.frame_stack-1]).mean()
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
            batch["action"][:, self.net.frame_stack-1],
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

