from typing import Literal, Optional, Tuple, Union

import torch
from gymnasium import spaces
from torch import Tensor, nn
from torch.distributions import Distribution

from ifo.common.nets.actor import ActionHead
from ifo.common.nets.base import FullyConnectedNeuralNetwork
from ifo.common.nets.impala_cnn import ImpalaCNNBackbone
from ifo.common.nets.quantizer import FiniteScalarQuantizer, IdentityQuantizer, VectorQuantizerEMA
from ifo.common.nets.u_net import UNet
from ifo.common.nets.world_model import ImpalaWorldModel, UNetWorldModel
from ifo.common.utils.functions import merge_tc, orthogonal_init


class SLAPOSegmentationNet(nn.Module):
    """UNet-based segmentation network used in SLAPO stage_0.

    This model predicts a binary foreground mask from a single observation frame.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        base_channels: int = 24,
    ) -> None:
        """Instantiate UNet-based segmentation network.

        Args:
            observation_space (spaces.Box): Observation space with shape (T, C, H, W).
            base_channels (int, optional): Base number of channels for the UNet backbone.
        """
        super().__init__()
        assert len(observation_space.shape) == 4, "Observation space must be 4-dimensional (T, C, H, W)."
        _, c, h, w = observation_space.shape
        self.obs_channels = c

        self.unet = UNet(
            shape=(c, h, w),
            output_channels=1,
            base_channels=base_channels,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x (Tensor): Input tensor of shape (B, T, C, H, W).

        Returns:
            Tensor: Predicted segmentation mask of shape (B, T, 1, H, W) with values in [0, 1].
        """
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        logits = self.unet(x)
        logits = logits.view(b, t, 1, h, w)
        return torch.sigmoid(logits)

    def label(self, x: Tensor, threshold: float = 0.5) -> Tensor:
        """Label with segmentation mask.

        Args:
            x (Tensor): Input tensor of shape (B, T, C, H, W).
            threshold (float, optional): Threshold for binarizing the mask. Defaults to 0.5.
        Returns:
            Tensor: Segmentation mask of shape (B, T, H, W) with values in {0, 1}.
        """
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        logits = self.unet(x)
        logits = logits.view(b, t, 1, h, w)
        mask = torch.sigmoid(logits) > threshold
        return mask


class SLAPOIDM(nn.Module):
    """Inverse dynamics model (IDM) class that encapsulates all NNs to train the IDM in SLAPO."""

    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Space,
        channel_multiplier: int,
        quantizer: Union[VectorQuantizerEMA, FiniteScalarQuantizer, IdentityQuantizer],
        world_model: Union[UNetWorldModel, ImpalaWorldModel],
        code_dim: int,
        num_latents: int,
        use_orthogonal_init: bool = False,
        future_obs_offset: int = 10,
        frame_stack: int = 3,
        add_mask_to_observation: bool = False,
        future_obs_sampling: bool = False,
        segmentation_net: Optional[SLAPOSegmentationNet] = None,
    ) -> None:
        """Instantiate LAPOIDM.

        Args:
            observation_space (spaces.Box): The observation space of the environment.
            action_space (spaces.Space): The action space of the environment.
            channel_multiplier (int): Channel multiplier of ImpalaCNN encoder.
            quantizer (Union[VectorQuantizerEMA, FiniteScalarQuantizer, IdentityQuantizer]): Quantizer to use.
            code_dim (int): Code dimension of VQ-VAE.
            num_latents (int): Number of discrete latents of VQ-VAE.
            use_orthogonal_init (bool): Whether to use orthogonal initialization for the model.
            future_obs_offset (int): Offset for the future observations.
            frame_stack (int): Frame stack for the observations.
            add_mask_to_observation (bool): Whether to add the mask to the observation as an additional channel.
            future_obs_sampling (bool): Whether to sample the future observations in an interval of 1 to
                future_obs_offset, if disabled, the future observations are the last frame_stack frames,
                k = future_obs_offset.
            segmentation_net (Optional[SLAPOSegmentationNet]): Segmentation network to use. If None, the ground-truth
                masks are used.
        """
        super().__init__()
        assert (
            code_dim % num_latents == 0
        ), f"Code dimension must be divisible by the number of latents, but is {code_dim} % {num_latents}."
        assert len(observation_space.shape) == 4, "Observation space must be 4-dimensional (T, C, H, W)."
        t, c, h, w = observation_space.shape
        assert t == future_obs_offset + frame_stack, f"""Future observation offset and frame stack must sum to the
        sequence length, but is {t} != {future_obs_offset} + {frame_stack}."""
        self.obs_channels = c
        self.code_dim = code_dim
        self.num_latents = num_latents
        c_encoder = c if not add_mask_to_observation else c + 1
        self.encoder = ImpalaCNNBackbone((2 * frame_stack * c_encoder, h, w), channel_multiplier, code_dim)
        self.decoder = world_model((frame_stack * c_encoder, h, w), output_channels=c, condition_dim=code_dim)
        self.information_bottleneck = quantizer
        self.action_decoder = ActionHead(code_dim, action_space, box_head_distribution="mse")
        self.future_obs_offset = future_obs_offset
        self.frame_stack = frame_stack
        self.add_mask_to_observation = add_mask_to_observation
        self.future_obs_sampling = future_obs_sampling
        self.segmentation_net = segmentation_net(observation_space=observation_space) if segmentation_net else None

        if use_orthogonal_init:
            self.apply(orthogonal_init)

    def _encode_action(self, observation: Tensor) -> Tuple[Tensor, Tensor, Tensor, Distribution]:
        """Shared IDM computation: (already mask-catted) observation -> latent action + action head.

        Factored out of `forward` so subclasses (``IMLAMIDM``) can reuse the exact quantizer /
        action-decoder logic while swapping only the FDM call. Returns the quantized latent action, the
        VQ loss, the perplexity, and the action distribution.
        """
        current_observation = merge_tc(observation[:, :self.frame_stack])

        # Sample the future observations in an interval of 1 to future_obs_offset, if enabled.
        # If disabled, the future observations are the last frame_stack frames, k = future_obs_offset.
        if self.future_obs_sampling:
            b = observation.shape[0]
            k = torch.randint(1, self.future_obs_offset, (b,), device=observation.device)
            frame_idx = k[:, None] + torch.arange(self.frame_stack, device=observation.device)
            future_observation = merge_tc(
                observation[torch.arange(b, device=observation.device)[:, None], frame_idx]
            )
        else:
            future_observation = merge_tc(observation[:, -self.frame_stack:])

        latent_action = self.encoder(torch.cat([current_observation, future_observation], dim=1))

        # Reshape in order to get multiple discrete latents, which are quantized independently.
        # (B, C) -> (B, C // num_latents, num_latents)
        b, c = latent_action.shape
        factorized_latent_action = latent_action.view(b, c // self.num_latents, self.num_latents)

        # Quantize the latents and convert back.
        quantized_latent_action, vq_loss, perplexity = self.information_bottleneck(factorized_latent_action)
        quantized_latent_action = quantized_latent_action.view(b, c)

        # Predict the action a_t given the quantized latent action.
        action_distribution = self.action_decoder(quantized_latent_action.detach())
        return quantized_latent_action, vq_loss, perplexity, action_distribution

    def forward(
        self, x: Tensor, mask: Optional[Tensor] = None, object_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Distribution, Tensor, Tensor]:
        """
        Forward pass.

        Args:
            x (Tensor): Sequence of observations from o_{t-frame_stack+1} ... o_{t+future_obs_offset}.
            mask (Optional[Tensor]): Mask of the observations.
            object_mask (Optional[Tensor]): Object mask; ignored here (MaskLAM/Foreground use only the
                agent mask), accepted so the shared module can pass it unconditionally to any net.
        Returns:
            Tuple[Tensor, Distribution, Tensor, Tensor]: Predicted next observation, action distribution,
                VQ loss and perplexity.
        """
        if self.add_mask_to_observation and mask is not None:
            observation = torch.cat([x, mask], dim=-3)
        else:
            observation = x

        current_observation = merge_tc(observation[:, :self.frame_stack])
        quantized_latent_action, vq_loss, perplexity, action_distribution = self._encode_action(observation)

        # Predict the next observation o_{t+1} given the current observation.
        next_observation = self.decoder(current_observation, quantized_latent_action)
        return next_observation, action_distribution, vq_loss, perplexity

    def label(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Label with latent action.

        Args:
            x (Tensor): Sequence of observations from o_{t-frame_stack+1} ... o_{t+future_obs_offset}.
            mask (Optional[Tensor]): Mask of the observations.
        Returns:
            Tensor: Latent action.
        """
        if self.add_mask_to_observation and mask is not None:
            observation = torch.cat([x, mask], dim=-3)
        else:
            observation = x
        current_observation = merge_tc(observation[:, :self.frame_stack])
        if self.future_obs_sampling:
            future_observation = merge_tc(observation[:, 1:self.frame_stack+1])
        else:
            future_observation = merge_tc(observation[:, -self.frame_stack:])
        return self.encoder(torch.cat([current_observation, future_observation], dim=1))


class IMLAMIDM(SLAPOIDM):
    """IM-LAM IDM: MaskLAM's IDM with the FDM replaced by the directed interaction predictor.

    Everything about the IDM (``self.encoder`` and the latent-action / quantizer / action-head path)
    is inherited unchanged from :class:`SLAPOIDM`, so a MaskLAM Stage-1 checkpoint's ``net.encoder.*``
    keys still load into Stage 2/3 (checkpoint contract). The only differences are:

    - **The FDM is the interaction world model, rebuilt at ``c+2`` input channels** (RGB + agent mask +
      object mask). ``super().__init__`` first builds the standard IDM and a *throwaway* FDM at the
      IDM's channel count; this class then re-invokes the same ``world_model`` partial at ``c+2`` and
      discards the throwaway. ``use_orthogonal_init`` is threaded into that rebuild explicitly, because
      the ``net.world_model`` Hydra partial does not carry it and ``super().__init__``'s
      ``apply(orthogonal_init)`` only reached the throwaway decoder.
    - **The forward routes both masks**: the FDM consumes the ``c+2`` current frame stack plus the
      current-frame (``frame_stack-1``) agent/object occupancy as attention / write-back signals.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Space,
        channel_multiplier: int,
        quantizer: Union[VectorQuantizerEMA, FiniteScalarQuantizer, IdentityQuantizer],
        world_model,
        code_dim: int,
        num_latents: int,
        use_orthogonal_init: bool = False,
        future_obs_offset: int = 10,
        frame_stack: int = 3,
        add_mask_to_observation: bool = False,
        future_obs_sampling: bool = False,
        segmentation_net: Optional[SLAPOSegmentationNet] = None,
        fdm_object_mask_input: bool = True,
    ) -> None:
        """Instantiate IMLAMIDM. Signature mirrors :class:`SLAPOIDM` plus ``fdm_object_mask_input``.

        Args:
            fdm_object_mask_input (bool): Documents the FDM asymmetry - the FDM (unlike the IDM) ingests
                the object mask as an extra input channel. The interaction module is defined around it
                (it routes the attention and the write-back gate), so there is no agent-only FDM variant:
                the only supported value is True, and False raises rather than silently doing nothing.
            (All other args: see :class:`SLAPOIDM`.)
        """
        if not fdm_object_mask_input:
            raise ValueError(
                "fdm_object_mask_input=False is not supported: the IM-LAM FDM is defined over the object "
                "mask (it drives the interaction attention and the write-back gate). Leave it True or drop "
                "the override."
            )
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            channel_multiplier=channel_multiplier,
            quantizer=quantizer,
            world_model=world_model,
            code_dim=code_dim,
            num_latents=num_latents,
            use_orthogonal_init=use_orthogonal_init,
            future_obs_offset=future_obs_offset,
            frame_stack=frame_stack,
            add_mask_to_observation=add_mask_to_observation,
            future_obs_sampling=future_obs_sampling,
            segmentation_net=segmentation_net,
        )
        self.fdm_object_mask_input = fdm_object_mask_input

        # Replace the throwaway FDM with the interaction world model at c+2 (RGB + agent + object mask).
        # use_orthogonal_init MUST be threaded in here: the partial does not carry it, and super()'s
        # global apply(orthogonal_init) already ran over the (now discarded) throwaway decoder.
        _, c, h, w = observation_space.shape
        self.decoder = world_model(
            (frame_stack * (c + 2), h, w),
            output_channels=c,
            condition_dim=code_dim,
            use_orthogonal_init=use_orthogonal_init,
        )

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        object_mask: Optional[Tensor] = None,
        agent_ctx_mode: str = "normal",
        return_attn: bool = False,
    ):
        """Forward pass.

        Args:
            x (Tensor): Sequence of observations ``(B, T, c, H, W)``.
            mask (Optional[Tensor]): Agent mask ``(B, T, 1, H, W)`` (required).
            object_mask (Optional[Tensor]): Object mask ``(B, T, 1, H, W)`` (required).
            agent_ctx_mode (str): Eval-time agent-path substitution passed to the interaction module.
            return_attn (bool): If True, append the object->agent attention maps as a 5th return value.
        Returns:
            The MaskLAM 4-tuple ``(next_observation, action_distribution, vq_loss, perplexity)``, plus a
            trailing attention tensor when ``return_attn`` is True.
        """
        if mask is None or object_mask is None:
            raise ValueError(
                "IMLAMIDM.forward requires both an agent mask and an object_mask (the FDM is defined "
                "over agent+object masks). Enable dataset.with_object_mask and route it through the module."
            )

        # IDM path: identical to MaskLAM (agent-mask-only, c+1) - reuse the shared quantizer/action logic.
        idm_observation = torch.cat([x, mask], dim=-3) if self.add_mask_to_observation else x
        quantized_latent_action, vq_loss, perplexity, action_distribution = self._encode_action(idm_observation)

        # FDM path: RGB + agent + object mask (c+2); current-frame (t = frame_stack-1) occupancy routes
        # the interaction attention and the write-back gate.
        fdm_observation = torch.cat([x, mask, object_mask], dim=-3)
        fdm_current = merge_tc(fdm_observation[:, :self.frame_stack])
        agent_mask_t = mask[:, self.frame_stack - 1]
        object_mask_t = object_mask[:, self.frame_stack - 1]
        fdm_out = self.decoder(
            fdm_current,
            quantized_latent_action,
            agent_mask_t,
            object_mask_t,
            agent_ctx_mode=agent_ctx_mode,
            return_attn=return_attn,
        )
        if return_attn:
            next_observation, attn = fdm_out
            return next_observation, action_distribution, vq_loss, perplexity, attn
        return fdm_out, action_distribution, vq_loss, perplexity

    def interaction_diagnostics(self) -> dict:
        """Scalar health signals for the interaction FDM, surfaced by the module from step 0.

        - ``beta_msa_a`` / ``beta_msa_o``: mean per-head mask-bias strength for the agent/object entity
          extractors. Should stay away from 0 - near-0 means the mask signal is being ignored.
        - ``beta_msa_a_min`` / ``beta_msa_o_min``: the *minimum* over heads. The N3 mask-collapse guard is
          per-head in spirit - a single head at 0 among five healthy ones would hide in the mean, so the
          min surfaces a collapsed head directly.
        - ``proj_a_norm`` / ``proj_o_norm``: write-back projection weight norms. Start at 0 (zero-init)
          and must grow; a norm stuck near 0 means the agent/object branch (and its ``z_t`` path) was
          made irrelevant. All detached, so this never perturbs the training graph.
        """
        interaction = self.decoder.interaction
        beta_a = interaction.msa_agent.beta.detach()
        beta_o = interaction.msa_object.beta.detach()
        return {
            "beta_msa_a": beta_a.mean(),
            "beta_msa_a_min": beta_a.min(),
            "beta_msa_o": beta_o.mean(),
            "beta_msa_o_min": beta_o.min(),
            "proj_a_norm": interaction.proj_a.weight.detach().norm(),
            "proj_o_norm": interaction.proj_o.weight.detach().norm(),
        }


class SLAPOLatentPolicy(nn.Module):
    """Latent policy class that encapsulates all NNs to train the latent policy in SLAPO."""

    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Space,
        channel_multiplier: int,
        code_dim: int,
        future_obs_offset: int = 10,
        frame_stack: int = 3,
        add_mask_to_observation: bool = False,
        mask_observation: bool = False,
        future_obs_sampling: bool = False,
        segmentation_net: Optional[SLAPOSegmentationNet] = None,
    ) -> None:
        """Instantiate latent policy.

        Args:
            observation_space (spaces.Box): The observation space of the environment.
            action_space (spaces.Space): The action space of the environment.
            channel_multiplier (int): Channel multiplier of ImpalaCNN encoder.
            code_dim (int): Code dimension of VQ-VAE respectively latent size of IDM.
            future_obs_offset (int): Offset for the future observations.
            frame_stack (int): Frame stack for the observations.
            add_mask_to_observation (bool): Whether to add the mask to the observation as an additional channel
                for the IDM.
            mask_observation (bool): Whether to mask the observations for the IDM.
            future_obs_sampling (bool): Whether the IDM was trained with future observation sampling.
            segmentation_net (Optional[SLAPOSegmentationNet]): Segmentation network to use. If None, the ground-truth
                masks are used.
        """
        super().__init__()
        assert len(observation_space.shape) == 4, "Observation space must be 4-dimensional (T, C, H, W)."
        t, c, h, w = observation_space.shape
        assert t == future_obs_offset + frame_stack, f"""Future observation offset and frame stack must sum to the
        sequence length, but is {t} != {future_obs_offset} + {frame_stack}."""
        self.obs_channels = c
        self.code_dim = code_dim
        self.add_mask_to_observation = add_mask_to_observation
        self.mask_observation = mask_observation
        self.future_obs_sampling = future_obs_sampling
        self.frame_stack = frame_stack
        c_encoder = c if not add_mask_to_observation else c + 1
        self.inverse_dynamics_model = ImpalaCNNBackbone((2 * frame_stack * c_encoder, h, w), channel_multiplier, code_dim)
        self.latent_policy = ImpalaCNNBackbone((frame_stack * c, h, w), channel_multiplier, code_dim)
        self.action_decoder = ActionHead(code_dim, action_space, box_head_distribution="mse")
        self.segmentation_net = segmentation_net(observation_space=observation_space) if segmentation_net else None

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Distribution]:
        """
        Forward pass.

        Args:
            x (Tensor): Sequence of observations from o_{t-frame_stack+1} ... o_{t+future_obs_offset}.
            mask (Optional[Tensor]): Mask of the observations.
        Returns:
            Tuple[Tensor, Tensor, Distribution]: Latent action, predicted latent action and action distribution.
        """
        # Replace ground-truth mask with predicted mask if segmentation model is available and needed.
        if self.segmentation_net is not None and (self.mask_observation or self.add_mask_to_observation):
            with torch.no_grad():
                self.segmentation_net.eval()
                mask = self.segmentation_net.label(x)

        # Mask the observations if the mask is provided and enabled.
        idm_observation = x
        if self.mask_observation and mask is not None:
            idm_observation = idm_observation * mask

        # Add mask to observation if enabled.
        if self.add_mask_to_observation and mask is not None:
            idm_observation = torch.cat([idm_observation, mask], dim=-3)

        idm_current_observation = merge_tc(idm_observation[:, :self.frame_stack])

        if self.future_obs_sampling:
            idm_future_observation = merge_tc(idm_observation[:, 1:self.frame_stack+1])
        else:
            idm_future_observation = merge_tc(idm_observation[:, -self.frame_stack:])

        with torch.no_grad():
            self.inverse_dynamics_model.eval()
            latent_action = self.inverse_dynamics_model(
                torch.cat([idm_current_observation, idm_future_observation], dim=1)
                )
        predicted_latent_action = self.latent_policy(merge_tc(x[:, :self.frame_stack]))
        action_distribution = self.action_decoder(predicted_latent_action.detach())
        return latent_action, predicted_latent_action, action_distribution

    def label(self, x: Tensor) -> Tensor:
        """Label with latent action.

        Args:
            x (Tensor): Sequence of observations from o_{t-k} ... o_{t+1}.

        Returns:
            Tensor: Latent action.
        """
        x = merge_tc(x[:, :self.frame_stack])
        return self.latent_policy(x)


class SLAPOBCActorAgent(nn.Module):
    """
    Actor class for behavior cloning fine-tuning of SLAPO.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Space,
        channel_multiplier: int,
        code_dim: int,
        hidden_dim: int = 256,
        box_head_distribution: Literal["normal", "tanh_normal", "scaled_normal", "mse"] = "scaled_normal",
        categorical_head_distribution: Literal["categorical", "one_hot_categorical"] = "categorical",
    ) -> None:
        """
        Initializes the SLAPOBCActorAgent.

        Args:
            observation_space (spaces.Box): The observation space of the environment.
            action_space (spaces.Space): The action space of the environment.
            channel_multiplier (int): Multiplier for the number of channels in the Impala feature extractor.
            code_dim (int): Dimension of the latent action.
            hidden_dim (int): Dimension of the hidden layers for the action decoder.
            box_head_distribution (Literal["normal", "tanh_normal", "scaled_normal", "mse"]): The distribution to
                use for the box head.
            categorical_head_distribution (Literal["categorical", "one_hot_categorical"]): The distribution to use
                for the categorical head.
        """
        super().__init__()
        assert len(observation_space.shape) == 4, "Observation space must be 4-dimensional."
        t, c, h, w = observation_space.shape
        self.obs_channels = c
        self.latent_policy = ImpalaCNNBackbone((t * c, h, w), channel_multiplier, code_dim)
        self.hidden_dim = hidden_dim
        self.actor = nn.Sequential(
            nn.ReLU(),
            nn.Linear(code_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            ActionHead(
                self.hidden_dim,
                action_space,
                box_head_distribution=box_head_distribution,
                categorical_head_distribution=categorical_head_distribution,
            ),
        )

    def forward(self, x: Tensor) -> Distribution:
        """Forward pass.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Distribution: Action distribution.
        """
        x = merge_tc(x)
        with torch.no_grad():
            self.latent_policy.eval()
            x = self.latent_policy(x)
        return self.actor(x)

