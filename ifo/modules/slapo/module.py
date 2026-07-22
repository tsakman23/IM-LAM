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
from ifo.modules.slapo.utils import (
    agent_occlusion,
    area_normalized_masked_mse,
    dilate_mask,
    dual_loss_grad_norms,
    dual_masked_reconstruction_loss,
    erode_mask,
)


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
        object_dual_loss: bool = False,
        object_loss_weight: float = 1.0,
        log_dual_loss_grad_every: int = 0,
        log_prefix: Optional[str] = None,
        action_variance: Optional[float] = None,
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
            object_dual_loss (bool, optional): If True, replace the single
                union-gated reconstruction loss with two separately area-normalized
                terms, L = L_A + object_loss_weight * L_O (requires an
                ``object_mask`` in the batch). Unlike ``object_mask_loss``'s union,
                each term gets its own normalization, so a small object mask isn't
                outweighed by the much larger agent mask. Mutually exclusive with
                ``object_mask_loss``. Defaults to False.
            object_loss_weight (float, optional): Weight ``lambda_o`` on the object
                term when ``object_dual_loss`` is True. Defaults to 1.0.
            log_dual_loss_grad_every (int, optional): If > 0 (and ``object_dual_loss``),
                every this many training steps, log the per-term gradient L2 norms
                of ``loss_agent`` and ``loss_object`` w.r.t. the shared FDM decoder.
                Area normalization alone does not guarantee comparable gradient
                influence on shared parameters - this checks it directly, at the
                cost of two extra backward passes every ``log_dual_loss_grad_every``
                steps. Training-only. 0 (default) disables the diagnostic entirely.
            log_prefix (Optional[str], optional): Stage scope to prepend to the
                image-panel keys logged by ``_log_debug_images``/``_log_debug_masks``/
                ``_log_debug_union_mask`` (e.g. ``"stage_1"``), matching the
                ``{log_prefix}/{key}`` scheme the Trainer applies to scalar metrics
                (``ifo.common.trainer.LightningTrainer._log``). These three methods
                call ``self.logger.experiment.log`` directly instead of returning
                through the Trainer, so without this they log unscoped keys
                (``train/pred_next_obs`` instead of ``stage_1/train/pred_next_obs``).
                Defaults to None (unscoped, e.g. for standalone/non-pipeline use).
            action_variance (Optional[float], optional): Per-task mean per-dimension
                variance of the clipped ground-truth actions, Var(a) in the MaskLAM/LAOM
                NMSE convention (Eq. 5). When set, ``action_decoder_nmse =
                action_decoder_mse / action_variance`` is additionally logged next to
                the existing (unnormalized) ``action_decoder_mse``. Defaults to None
                (metric not logged - this is an online, gradient-isolated probe trained
                jointly with Stage 1, not the paper's strict protocol of a probe
                retrained from scratch on a frozen checkpoint; treat NMSE values from
                this online probe as consistent for comparing your own arms against
                each other, not as directly comparable to MaskLAM's published numbers).
        """
        if object_mask_loss and object_dual_loss:
            raise ValueError(
                "object_mask_loss (union loss) and object_dual_loss "
                "(separately-normalized dual loss) are mutually exclusive ways to "
                "use the object mask in the loss - enable only one."
            )
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
        # separately-normalized agent/object dual loss (see docstring above).
        self.object_dual_loss = object_dual_loss
        self.object_loss_weight = object_loss_weight
        self.log_dual_loss_grad_every = log_dual_loss_grad_every
        self.log_prefix = log_prefix
        self.action_variance = action_variance

    def _scoped_key(self, key: str) -> str:
        """Prepend ``self.log_prefix`` (e.g. ``"stage_1"``) to a raw ``self.logger.experiment.log`` key.

        Only needed for the ``_log_debug_*`` methods below: they bypass the Trainer's
        ``_log`` (which applies this same scoping to every scalar metric automatically)
        and call the W&B run directly, so without this they log unscoped.
        """
        return f"{self.log_prefix}/{key}" if self.log_prefix else key

    @torch.no_grad()
    @torch.compiler.disable()
    def _log_debug_images(self, predicted_next_observation: Tensor, next_observation: Tensor, log_prefix: str) -> None:
        assert self.debug_transform is not None, "Debug transform is not provided"
        pred_next_obs = self.debug_transform(predicted_next_observation)
        gt_next_obs = self.debug_transform(next_observation)
        self.logger.experiment.log(
            {
                self._scoped_key(f"{log_prefix}/pred_next_obs"): wandb.Image(
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
                self._scoped_key(f"{log_prefix}/mask"): wandb.Image(make_comp_grid(mask * 255.0, gt_mask * 255.0), caption="Modified mask vs ground truth"),
            },
        )

    @torch.no_grad()
    @torch.compiler.disable()
    def _log_debug_union_mask(
        self, union_mask: Tensor, agent_mask: Tensor, next_observation: Tensor, log_prefix: str
    ) -> None:
        """Foreground-MaskLAM: log the agent U object union mask actually used.

        Only called on Foreground-MaskLAM runs (``object_mask_loss`` or ``object_mask_input``);
        stock MaskLAM logging is untouched. Two panels are emitted under ``{prefix}``:

        - ``union_mask``: the union (top) vs the agent-only mask (bottom) as a white-on-black
          comp grid, so the manipulated-object pixels the union adds are directly visible.
        - ``union_mask_overlay``: the union tinted green over the (un-centered) ground-truth
          next observation, so the union can be read against the scene it gates.

        Args:
            union_mask (Tensor): Union mask of shape (B, 1, H, W) in {0, 1}.
            agent_mask (Tensor): Agent-only mask of shape (B, 1, H, W) in {0, 1}.
            next_observation (Tensor): Centered GT next observation of shape (B, C, H, W).
            log_prefix (str): Prefix for logging.
        """
        assert self.debug_transform is not None, "Debug transform is not provided"
        gt_next_obs = self.debug_transform(next_observation)  # (B, C, H, W) in [0, 255]
        # Green tint on the union region, blended over the observation.
        alpha = 0.5
        green = torch.zeros_like(gt_next_obs)
        green[:, 1] = 255.0
        overlay = gt_next_obs * (1.0 - alpha * union_mask) + green * (alpha * union_mask)
        self.logger.experiment.log(
            {
                self._scoped_key(f"{log_prefix}/union_mask"): wandb.Image(
                    make_comp_grid(union_mask * 255.0, agent_mask * 255.0),
                    caption="Union (agent U object) mask vs agent-only mask",
                ),
                self._scoped_key(f"{log_prefix}/union_mask_overlay"): wandb.Image(
                    make_comp_grid(overlay, gt_next_obs),
                    caption="Union mask (green) over next obs vs next obs",
                ),
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

    @torch.compiler.disable()
    def _dual_loss_grad_diagnostics(self, loss_agent: Tensor, loss_object: Tensor) -> tuple[Tensor, Tensor]:
        """Periodic diagnostic for ``log_dual_loss_grad_every``: per-term gradient norms w.r.t. the shared decoder.

        Split out of ``_forward`` and disabled for ``torch.compile`` (``training_step`` is compiled
        end-to-end when ``trainer.compile=True``, and ``torch.autograd.grad`` cannot run inside a
        compiled graph). Called with un-compiled eager execution instead, same as the other
        ``_log_debug_*`` helpers below.

        Args:
            loss_agent (Tensor): Scalar agent-region loss term.
            loss_object (Tensor): Scalar object-region loss term.

        Returns:
            Tuple of ``(grad_norm_agent, grad_norm_object)`` scalar tensors.
        """
        decoder_params = [p for p in self.net.decoder.parameters() if p.requires_grad]
        return dual_loss_grad_norms(loss_agent, loss_object, decoder_params)

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
        # Fail fast: a Foreground-MaskLAM flag enabled without object masks in the batch
        # would otherwise silently fall back to agent-only masks and train plain MaskLAM
        # under a Foreground-MaskLAM config, corrupting the ablation.
        if (self.object_mask_loss or self.object_mask_input or self.object_dual_loss) and object_mask is None:
            raise ValueError(
                "object_mask_loss/object_mask_input/object_dual_loss is enabled but the batch has "
                "no 'object_mask'. Load object masks with dataset.with_object_mask=true and an "
                "object-mask repo (e.g. dataset.dataset_path=tsakman23/visual_masked_distracting_metaworld), "
                "or disable the object-mask flags to run plain MaskLAM."
            )
        has_object = object_mask is not None
        input_mask = agent_mask
        loss_mask = agent_mask
        foreground_mask = None
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

        loss_agent = loss_object = None
        grad_norm_agent = grad_norm_object = None
        if self.mask_loss and agent_mask is not None:
            error = F.mse_loss(
                next_observation, batch["observation"][:, self.net.frame_stack], reduction="none")
            if self.object_dual_loss and has_object:
                # Dual: two separately-normalized terms instead of one union-gated term.
                agent_mask_t = agent_mask[:, self.net.frame_stack]
                object_mask_t = object_mask[:, self.net.frame_stack]
                reconstruction_loss, loss_agent, loss_object = dual_masked_reconstruction_loss(
                    error, agent_mask_t, object_mask_t, self.object_loss_weight
                )
                # Periodic diagnostic: is L_O actually moving the shared decoder,
                # or is area normalization not enough to keep it from being
                # drowned out? Training-only (validation runs under no_grad).
                if (
                    self.log_dual_loss_grad_every > 0
                    and prefix == "train"
                    and self.global_step % self.log_dual_loss_grad_every == 0
                ):
                    grad_norm_agent, grad_norm_object = self._dual_loss_grad_diagnostics(
                        loss_agent, loss_object
                    )
            else:
                mask = loss_mask[:, self.net.frame_stack]
                reconstruction_loss = area_normalized_masked_mse(error, mask)
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
        if self.action_variance is not None and "action_decoder_mse" in step_dict.keys():
            step_dict["action_decoder_nmse"] = step_dict["action_decoder_mse"] / self.action_variance
        if self.dilate_mask or self.erode_mask:
            step_dict["miou"] = miou
        if loss_agent is not None:
            step_dict["reconstruction_loss_agent"] = loss_agent
            step_dict["reconstruction_loss_object"] = loss_object
        if grad_norm_agent is not None:
            step_dict["reconstruction_loss_agent_grad_norm"] = grad_norm_agent
            step_dict["reconstruction_loss_object_grad_norm"] = grad_norm_object

        # Log predicted next observation vs ground truth on the last batch.
        if batch_idx == batch_len - 1 and self.debug_transform is not None:
            self._log_debug_images(next_observation, batch["observation"][:, self.net.frame_stack], prefix)
            if self.dilate_mask or self.erode_mask:
                self._log_debug_masks(batch["mask"][:, self.net.frame_stack], original_mask[:, self.net.frame_stack], prefix)
            # Foreground-MaskLAM: additionally log the agent U object union actually used
            # to gate the FDM loss / mask input. Guarded so stock MaskLAM logging is untouched.
            if (self.object_mask_loss or self.object_mask_input) and foreground_mask is not None:
                self._log_debug_union_mask(
                    foreground_mask[:, self.net.frame_stack],
                    agent_mask[:, self.net.frame_stack],
                    batch["observation"][:, self.net.frame_stack],
                    prefix,
                )

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

