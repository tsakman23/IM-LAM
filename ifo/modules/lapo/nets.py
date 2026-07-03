from typing import List, Optional, Tuple, Union

import torch
from gymnasium import spaces
from torch import Tensor, nn
from torch.distributions import Categorical, Distribution, Independent, Normal

from ifo.common.distributions import DictDistribution, MultiCategorical, TruncatedNormal
from ifo.common.nets.actor import ActionHead
from ifo.common.nets.base import FullyConnectedNeuralNetwork
from ifo.common.nets.impala_cnn import ImpalaCNNBackbone
from ifo.common.nets.quantizer import FiniteScalarQuantizer, IdentityQuantizer, VectorQuantizerEMA
from ifo.common.nets.world_model import ImpalaWorldModel, UNetWorldModel
from ifo.common.utils.functions import merge_tc, orthogonal_init


class LAPOIDM(nn.Module):
    """Inverse dynamics model (IDM) class that encapsulates all NNs to train the IDM in LAPO."""

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
        """
        super().__init__()
        assert (
            code_dim % num_latents == 0
        ), f"Code dimension must be divisible by the number of latents, but is {code_dim} % {num_latents}."
        assert len(observation_space.shape) == 4, "Observation space must be 4-dimensional (T, C, H, W)."
        t, c, h, w = observation_space.shape
        self.obs_channels = c
        self.code_dim = code_dim
        self.num_latents = num_latents
        self.encoder = ImpalaCNNBackbone((t * c, h, w), channel_multiplier, code_dim)
        self.decoder = world_model(((t - 1) * c, h, w), output_channels=c, condition_dim=code_dim)
        self.information_bottleneck = quantizer
        self.action_decoder = ActionHead(code_dim, action_space, box_head_distribution="mse")

        if use_orthogonal_init:
            self.apply(orthogonal_init)

    def forward(self, x: Tensor) -> Tuple[Tensor, Distribution, Tensor, Tensor]:
        """
        Forward pass.

        Args:
            x (Tensor): Sequence of observations from o_{t-k} ... o_{t+1}.

        Returns:
            Tuple[Tensor, Distribution, Tensor, Tensor]: Predicted next observation, action distribution,
                VQ loss and perplexity.
        """
        x = merge_tc(x)
        latent_action = self.encoder(x)

        # Reshape in order to get multiple discrete latents, which are quantized independently.
        # (B, C) -> (B, C // num_latents, num_latents)
        b, c = latent_action.shape
        factorized_latent_action = latent_action.view(b, c // self.num_latents, self.num_latents)

        # Quantize the latents and convert back.
        quantized_latent_action, vq_loss, perplexity = self.information_bottleneck(factorized_latent_action)
        quantized_latent_action = quantized_latent_action.view(b, c)

        # Predict the action a_t given the quantized latent action.
        action_distribution = self.action_decoder(quantized_latent_action.detach())

        # Predict the next observation o_{t+1} given the sequence of observations o_{t-k} ... o_{t}.
        next_observation = self.decoder(x[:, : -self.obs_channels], quantized_latent_action)
        return next_observation, action_distribution, vq_loss, perplexity

    def label(self, x: Tensor) -> Tensor:
        """Label with latent action.

        Args:
            x (Tensor): Sequence of observations from o_{t-k} ... o_{t+1}.

        Returns:
            Tensor: Latent action.
        """
        x = merge_tc(x)
        return self.encoder(x)


class LAPOLatentActionDecoder(nn.Module):
    """Decoder for the latent action."""

    def __init__(
        self, in_features: int, out_features: int, hidden_layers: List[int], action_space: spaces.Space
    ) -> None:
        """Instantiate the latent action decoder.

        Args:
            in_features (int): The number of features of the input.
            out_features (int): The number of features of the output.
            hidden_layers (List[int]): The number of features of the hidden layers.
            action_space (spaces.Space): The action space of the environment.
        """
        super().__init__()
        self.decoder = nn.Sequential(
            FullyConnectedNeuralNetwork(in_features, out_features, hidden_layers),
            ActionHead(out_features, action_space, box_head_distribution="mse"),
        )

    def forward(self, x: Tensor) -> Distribution:
        """Forward pass.

        Args:
            x (Tensor): The input tensor with latent action.

        Returns:
            Distribution: The action distribution.
        """
        return self.decoder(x)


class LAPOLatentPolicy(nn.Module):
    """Latent policy class that encapsulates all NNs to train the latent policy in LAPO."""

    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Space,
        channel_multiplier: int,
        code_dim: int,
        policy_block_size: Optional[int] = None,
    ) -> None:
        """Instantiate latent policy.

        Args:
            observation_space (spaces.Box): The observation space of the environment.
            action_space (spaces.Space): The action space of the environment.
            channel_multiplier (int): Channel multiplier of ImpalaCNN encoder.
            code_dim (int): Code dimension of VQ-VAE respectively latent size of IDM.
            policy_block_size (Optional[int], optional): Block size for the latent policy. If None, the block size is
                set to the sequence length minus 1.
        """
        super().__init__()
        assert len(observation_space.shape) == 4, "Observation space must be 4-dimensional (T, C, H, W)."
        t, c, h, w = observation_space.shape
        self.obs_channels = c
        self.code_dim = code_dim
        self.inverse_dynamics_model = ImpalaCNNBackbone((t * c, h, w), channel_multiplier, code_dim)

        self.policy_block_size = t - 1 if policy_block_size is None else policy_block_size
        assert self.policy_block_size > 0 and self.policy_block_size < t, "Block size must be in [1, t-1]."
        self.latent_policy = ImpalaCNNBackbone((self.policy_block_size * c, h, w), channel_multiplier, code_dim)
        self.start_dim = (t - 1 - self.policy_block_size) * c
        self.stop_dim = -1 * c
        self.action_decoder = ActionHead(code_dim, action_space, box_head_distribution="mse")

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Distribution]:
        """
        Forward pass.

        Args:
            x (Tensor): Sequence of observations from o_{t-k} ... o_{t+1}.

        Returns:
            Tuple[Tensor, Tensor, Distribution]: Latent action, predicted latent action and action distribution.
        """
        x = merge_tc(x)
        with torch.no_grad():
            self.inverse_dynamics_model.eval()
            latent_action = self.inverse_dynamics_model(x)
        predicted_latent_action = self.latent_policy(x[:, self.start_dim : self.stop_dim])
        action_distribution = self.action_decoder(predicted_latent_action.detach())
        return latent_action, predicted_latent_action, action_distribution

    def label(self, x: Tensor) -> Tensor:
        """Label with latent action.

        Args:
            x (Tensor): Sequence of observations from o_{t-k} ... o_{t+1}.

        Returns:
            Tensor: Latent action.
        """
        x = merge_tc(x)
        return self.latent_policy(x[:, self.start_dim : self.stop_dim])


class LAPOActorCriticAgent(nn.Module):
    """Policy class that encapsulates all NNs to train the policy in LAPO."""

    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Space,
        channel_multiplier: int,
        code_dim: int,
        idm_block_size: Optional[int] = None,
    ) -> None:
        """Instantiate policy.

        Args:
            observation_space (spaces.Box): The observation space of the environment.
            action_space (spaces.Space): The action space of the environment.
            channel_multiplier (int, optional): Multiplier for the number of channels in the Impala feature extractor.
            code_dim (int): Code dimension of VQ-VAE respectively latent size of IDM.
            idm_block_size (int, optional): Block size for the inverse dynamics model. If None, the block
                size is set to the block size of the observations plus one.
        """
        super().__init__()
        assert len(observation_space.shape) == 4, "Observation space must be 4-dimensional (T, C, H, W)."
        t, c, h, w = observation_space.shape
        self.obs_channels = c
        self.latent_policy = ImpalaCNNBackbone((t * c, h, w), channel_multiplier, code_dim)

        self.idm_block_size = t + 1 if idm_block_size is None else idm_block_size
        assert self.idm_block_size > t, "IDM block size must be in [t+1, inf]."
        self.inverse_dynamics_model = ImpalaCNNBackbone((self.idm_block_size * c, h, w), channel_multiplier, code_dim)

        # Clone of the last fc layer of the inverse dynamics model, which is trained during policy training.
        # The IDM and its fc layer are not trained during policy training.
        self.final_fc_rl = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.latent_policy.final_fc.in_features, self.latent_policy.final_fc.out_features),
        )

        # RL policy head.
        self.actor = nn.Sequential(nn.ReLU(), ActionHead(code_dim, action_space))

        # Decoder for supervised decoding of the latent actions.
        self.decoder = nn.Sequential(
            nn.ReLU(), FullyConnectedNeuralNetwork(code_dim, 64, [192, 128]), ActionHead(64, action_space)
        )

        self.critic = nn.Sequential(nn.ReLU(), nn.Linear(code_dim, 1))
        self.critic.apply(orthogonal_init)

    def clone_fc(self) -> None:
        """Clone the last fully connected layer of the inverse dynamics model."""
        self.final_fc_rl[-1].load_state_dict(self.latent_policy.final_fc.state_dict())

    def label(self, x: Tensor) -> Tensor:
        """
        Forward pass for action encoder.

        Args:
            x (Tensor): Sequence of observations from o_{t-k} ... o_{t+1}.

        Returns:
            Tensor: Latent action.
        """
        x = merge_tc(x)
        return self.inverse_dynamics_model(x)

    def merge_action_heads(self, rl_dist: Distribution, sl_dist: Distribution) -> Distribution:
        """Merge the action heads of the RL and SL policy.

        Note: This function detaches the SL distribution from the graph to avoid backpropagation through the SL policy
        and only train the RL policy.

        Args:
            rl_head (Distribution): The action distribution of the RL policy.
            sl_head (Distribution): The action distribution of the SL policy.
        """
        assert isinstance(sl_dist, type(rl_dist)), "RL and SL distributions must be of the same type."
        if isinstance(rl_dist, Independent):
            return Independent(
                self.merge_action_heads(rl_dist.base_dist, sl_dist.base_dist), rl_dist.reinterpreted_batch_ndims
            )
        if isinstance(rl_dist, (TruncatedNormal, Normal)):
            loc = (rl_dist.loc + sl_dist.loc.detach()) / 2
            scale = (rl_dist.scale + sl_dist.scale.detach()) / 2
            if isinstance(rl_dist, TruncatedNormal):
                return TruncatedNormal(loc, scale, rl_dist._non_std_a, rl_dist._non_std_b)
            elif isinstance(rl_dist, Normal):
                return Normal(loc, scale)
            else:
                raise NotImplementedError
        elif isinstance(rl_dist, Categorical):
            logits = (rl_dist.logits + sl_dist.logits.detach()) / 2
            return Categorical(logits=logits)
        elif isinstance(rl_dist, MultiCategorical):
            logits = [
                (rl.logits + sl.logits.detach()) / 2 for rl, sl in zip(rl_dist.distributions, sl_dist.distributions)
            ]
            return MultiCategorical(logits=logits)
        elif isinstance(rl_dist, DictDistribution):
            distributions = {
                key: self.merge_action_heads(rl_dist.distributions[key], sl_dist.distributions[key])
                for key in rl_dist.distributions.keys()
            }
            return DictDistribution(distributions)
        else:
            raise NotImplementedError

    def forward(self, x: Tensor) -> Tuple[Distribution, Tensor]:
        """Forward pass.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tuple[Distribution, Tensor]: A tuple containing the action distribution and critic value.
        """
        x = merge_tc(x)
        x = self.latent_policy.backbone(x)

        hidden_sl = self.latent_policy.final_fc(x)
        hidden_rl = self.final_fc_rl(x)

        # Average logits from SL and RL policy.
        # SL branch is only trained in supervised learning callback!
        sl_dist = self.decoder(hidden_sl)
        rl_dist = self.actor(hidden_rl)
        return self.merge_action_heads(rl_dist, sl_dist), self.critic(hidden_rl).view(-1)
