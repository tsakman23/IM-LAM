import os
from typing import Callable, Optional, Tuple

import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from tensordict import TensorDict
from torch import Tensor
from torch.distributions.kl import kl_divergence

import ifo.common.distributions as itd
from ifo.common.module import SupervisedLightningModule
from ifo.modules.dreamerv3.buffer import ReplayBuffer
from ifo.modules.dreamerv3.nets import DreamerV3ActorCriticAgent
from ifo.modules.dreamerv3.utils import Ratio, compute_lambda_returns


class DreamerV3Module(SupervisedLightningModule):
    """
    DreamerV3 module for reinforcement learning.

    This module implements the DreamerV3 algorithm for training reinforcement learning agents.
    """

    def __init__(
        self,
        net: nn.Module,
        batch_size: int,
        optimizer: DictConfig,
        lr_scheduler: Optional[DictConfig] = None,
        num_workers: int = 0,
        max_grad_norm: Optional[float] = None,
        block_size: int = 64,
        horizon: int = 15,
        replay_ratio: float = 1.0,
        replay_buffer_size: int = int(1e6),
        replay_buffer_memmap: bool = False,
        gamma: float = 0.996996996996997,
        lmbda: float = 0.95,
        continuous_discounting: bool = True,
        free_nats: float = 1.0,
        dynamics_scale_factor: float = 1.0,
        representation_scale_factor: float = 0.1,
        reconstruction_scale_factor: float = 1.0,
        reward_scale_factor: float = 1.0,
        actor_scale_factor: float = 1.0,
        critic_scale_factor: float = 1.0,
        continue_scale_factor: float = 1.0,
        entropy_scale: float = 3e-4,
        target_network_update_freq: int = 1,
        tau: float = 0.02,
        debug_transform: Optional[Callable] = None,
    ) -> None:
        """
        Initialize the DreamerV3 module.

        Args:
            net (nn.Module): The neural network model of the agent.
            batch_size (int): The batch size for training.
            optimizer (DictConfig): The optimizer to use for the DreamerV3 module.
            lr_scheduler (Optional[DictConfig], optional): The learning rate scheduler to use for the optimizer.
            num_workers (int, optional): The number of workers for the dataloader. Defaults to 0.
            max_grad_norm (Optional[float], optional): The maximum gradient norm for gradient clipping.
                Defaults to None.
            block_size (int, optional): The block size for sequence sampling. Defaults to 64.
            horizon (int, optional): The planning horizon for behavior learning. Defaults to 15.
            replay_ratio (float, optional): The replay ratio for training. Defaults to 1.0.
            replay_buffer_size (int, optional): The size of the replay buffer. Defaults to 1e6.
            replay_buffer_memmap (bool, optional): Whether to use memory-mapped replay buffer.
                Defaults to False.
            gamma (float, optional): The discount factor. Defaults to 0.996996996996997.
            lmbda (float, optional): The lambda parameter for GAE. Defaults to 0.95.
            continuous_discounting (bool, optional): Whether to use continuous discounting. Defaults to True.
            free_nats (float, optional): KL free bits threshold. Defaults to 1.0.
            dynamics_scale_factor (float, optional): Scale factor for dynamics loss. Defaults to 1.0.
            representation_scale_factor (float, optional): Scale factor for representation loss.
                Defaults to 0.1.
            reconstruction_scale_factor (float, optional): Scale factor for reconstruction loss.
                Defaults to 1.0.
            reward_scale_factor (float, optional): Scale factor for reward loss. Defaults to 1.0.
            actor_scale_factor (float, optional): Scale factor for actor loss. Defaults to 1.0.
            critic_scale_factor (float, optional): Scale factor for critic loss. Defaults to 1.0.
            continue_scale_factor (float, optional): Scale factor for continue loss. Defaults to 1.0.
            entropy_scale (float, optional): Entropy coefficient for exploration. Defaults to 3e-4.
            target_network_update_freq (int, optional): Frequency of target network updates. Defaults to 1.
            tau (float, optional): Soft update coefficient for target networks. Defaults to 0.02.
            debug_transform (Optional[Callable], optional): The debug transform to use for image visualization.
        """

        super().__init__(
            net=net,
            batch_size=batch_size,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            num_workers=num_workers,
            max_grad_norm=max_grad_norm,
        )

        self.block_size = block_size
        self.horizon = horizon
        self.replay_ratio = replay_ratio

        self.replay_buffer_size = int(replay_buffer_size)
        self.replay_buffer_memmap = replay_buffer_memmap
        self.gamma = gamma
        self.lmbda = lmbda
        self.continuous_discounting = continuous_discounting
        self.free_nats = free_nats
        self.dynamics_scale_factor = dynamics_scale_factor
        self.representation_scale_factor = representation_scale_factor
        self.reconstruction_scale_factor = reconstruction_scale_factor
        self.reward_scale_factor = reward_scale_factor
        self.actor_scale_factor = actor_scale_factor
        self.critic_scale_factor = critic_scale_factor
        self.continue_scale_factor = continue_scale_factor
        self.ent_coef = entropy_scale
        self.target_network_update_freq = target_network_update_freq
        self.tau = tau
        self.debug_transform = debug_transform

    def training_step(self, batch: TensorDict, batch_idx: int, batch_len: int) -> TensorDict:
        """Train for one step.

        Args:
            batch (TensorDict): Batch with observation and action.
            batch_idx (int): Index of batch.
            batch_len (int): Length of batches in dataloader.
        Returns:
            TensorDict: Dictionary with loss and other metrics to log.
        """
        # Prepare the batch.
        # Set the first is_first to 1 for the first step.
        batch["is_first"][0] = torch.ones_like(batch["is_first"][0])
        # According to the environment interaction, we remove the last actions and add the first one as the zero action
        batch["actions"] = torch.cat((torch.zeros_like(batch["actions"][:1]), batch["actions"][:-1]), dim=0)

        # Dynamics Learning
        (world_loss, kl, dyn_loss, repr_loss, reward_loss, observation_loss, continue_loss) = (
            self._training_step_dynamics(batch)
        )

        # Behaviour Learning
        policy_loss, value_loss, entropy_loss = self._training_step_behavior(batch)

        # Log metrics
        post_entropy = (
            td.Independent(td.OneHotCategorical(logits=batch["posteriors_logits"].detach()), 1).entropy().mean()
        )
        prior_entropy = td.Independent(td.OneHotCategorical(logits=batch["priors_logits"].detach()), 1).entropy().mean()
        return TensorDict(
            {
                "loss": world_loss + policy_loss + value_loss,
                "world_model_loss": world_loss,
                "entropy_loss": entropy_loss,
                "value_loss": value_loss,
                "policy_loss": policy_loss,
                "dyn_loss": dyn_loss,
                "repr_loss": repr_loss,
                "reward_loss": reward_loss,
                "observation_loss": observation_loss,
                "continue_loss": continue_loss,
                "kl": kl.mean(),
                "post_entropy": post_entropy,
                "prior_entropy": prior_entropy,
            }
        )

    def _training_step_dynamics(self, batch: TensorDict) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Train the dynamics model.

        This method trains the world model components including:
        - RSSM (Recurrent State Space Model) for state transitions
        - Encoder for observation embedding
        - Decoder for observation reconstruction
        - Reward model for reward prediction
        - Continue model for episode termination prediction

        Args:
            batch (TensorDict): Batch with observation, action, reward, terminated, is_first.

        Returns:
            Tuple containing:
            - world_model_loss: World model loss
            - kl: KL divergence between prior and posterior
            - dynamics_loss: Dynamics loss
            - representation_loss: Representation loss
            - reward_loss: Reward loss
            - observation_loss: Observation loss
            - continue_loss: Continue prediction loss
        """
        assert isinstance(self.net, DreamerV3ActorCriticAgent)

        # Initialize tensors for the dynamics model.
        recurrent_state = torch.zeros(1, self.batch_size, self.net.recurrent_state_size, device=batch.device)
        recurrent_states = torch.empty(
            self.block_size,
            self.batch_size,
            self.net.recurrent_state_size,
            device=batch.device,
        )
        priors_logits = torch.empty(
            self.block_size,
            self.batch_size,
            self.net.stochastic_state_size,
            device=batch.device,
        )
        posterior = torch.zeros(
            1,
            self.batch_size,
            self.net.stochastic_size,
            self.net.discrete_size,
            device=batch.device,
        )
        posteriors = torch.empty(
            self.block_size,
            self.batch_size,
            self.net.stochastic_size,
            self.net.discrete_size,
            device=batch.device,
        )
        posteriors_logits = torch.empty(
            self.block_size,
            self.batch_size,
            self.net.stochastic_state_size,
            device=batch.device,
        )

        # The environment interaction goes like this:
        # Actions:           a0       a1       a2      a3
        #                    ^ \      ^ \      ^ \     ^
        #                   /   \    /   \    /   \   /
        #                  /     v  /     v  /     v /
        # Observations:  o0       o1       o2       o3
        # Rewards:       0        r1       r2       r3
        # Dones:         0        d1       d2       d3
        # Is-first       1        i1       i2       i3

        # Embed observations from the environment
        embedded_obs = self.net.world_model.encoder(batch)

        # Compute the dynamics model.
        for i in range(0, self.block_size):
            recurrent_state, posterior, _, posterior_logits, prior_logits = self.net.world_model.rssm.dynamic(
                posterior,
                recurrent_state,
                batch["actions"][i : i + 1],
                embedded_obs[i : i + 1],
                batch["is_first"][i : i + 1],
            )
            recurrent_states[i] = recurrent_state
            priors_logits[i] = prior_logits
            posteriors[i] = posterior
            posteriors_logits[i] = posterior_logits
        latent_states = torch.cat((posteriors.view(*posteriors.shape[:-2], -1), recurrent_states), -1)

        # Compute predictions for the observations
        reconstructed_obs = self.net.world_model.decoder(latent_states)

        # Compute observation loss
        predicted_observations_dist = {}
        for k in self.net.cnn_keys:
            predicted_observations_dist[k] = itd.MSEDistribution(
                reconstructed_obs[k], dims=len(reconstructed_obs[k].shape[2:])
            )
        for k in self.net.mlp_keys:
            predicted_observations_dist[k] = itd.SymlogDistribution(
                reconstructed_obs[k], dims=len(reconstructed_obs[k].shape[2:])
            )
        predicted_observations_dist = TensorDict(
            predicted_observations_dist, batch_size=latent_states.shape[:-1], device=reconstructed_obs.device
        )
        observation_loss = -sum(
            [predicted_observations_dist[k].log_prob(batch[k]).mean() for k in predicted_observations_dist.keys()]
        )

        # Compute the distribution over the rewards
        predicted_rewards_dist = itd.TwoHotEncodingDistribution(
            self.net.world_model.reward_model(latent_states), dims=1
        )

        # Compute the distribution over the terminal steps, if required
        predicted_continues_dist = td.Independent(
            itd.BernoulliSafe(logits=self.net.world_model.continue_model(latent_states)), 1
        )

        # Reshape posterior and prior logits to shape [B, T, 32, 32]
        priors_logits = priors_logits.view(*priors_logits.shape[:-1], self.net.stochastic_size, self.net.discrete_size)
        posteriors_logits = posteriors_logits.view(
            *posteriors_logits.shape[:-1], self.net.stochastic_size, self.net.discrete_size
        )

        # KL divergence between prior and posterior
        dynamics_loss = kl = kl_divergence(
            td.Independent(td.OneHotCategoricalStraightThrough(logits=posteriors_logits.detach()), 1),
            td.Independent(td.OneHotCategoricalStraightThrough(logits=priors_logits), 1),
        )
        representation_loss = kl_divergence(
            td.Independent(td.OneHotCategoricalStraightThrough(logits=posteriors_logits), 1),
            td.Independent(td.OneHotCategoricalStraightThrough(logits=priors_logits.detach()), 1),
        )

        # Regularize the KL divergence with free nats term
        free_nats = torch.full_like(dynamics_loss, self.free_nats)
        dynamics_loss = torch.maximum(dynamics_loss, free_nats).mean()
        representation_loss = torch.maximum(representation_loss, free_nats).mean()

        reward_loss = -predicted_rewards_dist.log_prob(batch["rewards"]).mean()

        continue_targets = 1 - batch["terminated"]
        if self.continuous_discounting:
            continue_targets *= self.gamma
        continue_loss = -predicted_continues_dist.log_prob(continue_targets).mean()

        world_model_loss = (
            self.dynamics_scale_factor * dynamics_loss
            + self.representation_scale_factor * representation_loss
            + self.reconstruction_scale_factor * observation_loss
            + self.reward_scale_factor * reward_loss
            + self.continue_scale_factor * continue_loss
        )

        # Add posteriors and recurrent states to the batch
        batch["posteriors"] = posteriors
        batch["recurrent_states"] = recurrent_states
        batch["posteriors_logits"] = posteriors_logits
        batch["priors_logits"] = priors_logits

        return (
            world_model_loss,
            kl.mean(),
            dynamics_loss,
            representation_loss,
            reward_loss,
            observation_loss,
            continue_loss,
        )

    def _training_step_behavior(self, batch: TensorDict) -> Tuple[Tensor, Tensor, Tensor]:
        """Train the behaviour model.

        This method trains the actor and critic models.

        Args:
            batch (TensorDict): Batch with observation, action, reward, terminated, is_first.

        Returns:
            Tuple containing:
            - policy_loss: Policy loss
            - value_loss: Value loss
            - entropy_loss: Entropy loss
        """
        imagined_prior = batch["posteriors"].reshape(1, -1, self.net.stochastic_state_size)
        recurrent_state = batch["recurrent_states"].reshape(1, -1, self.net.recurrent_state_size)
        imagined_latent_state = torch.cat((imagined_prior, recurrent_state), -1)
        imagined_trajectories = torch.empty(
            self.horizon + 1,
            self.batch_size * self.block_size,
            self.net.stochastic_state_size + self.net.recurrent_state_size,
            device=batch.device,
        )
        imagined_trajectories[0] = imagined_latent_state  # Last real state
        imagined_actions = torch.empty(
            self.horizon + 1,
            self.batch_size * self.block_size,
            batch["actions"].shape[-1],
            device=batch.device,
        )
        imagined_actions[0] = actions = self.net.actor(imagined_latent_state.detach()).rsample()

        # Actor optimization step. Eq. 11 from the paper
        # The imagination goes like this, with H=3:
        # Actions:           a'0      a'1      a'2     a'3
        #                    ^ \      ^ \      ^ \     ^
        #                   /   \    /   \    /   \   /
        #                  /     \  /     \  /     \ /
        # States:        z0 ---> z'1 ---> z'2 ---> z'3
        # Rewards:       r'0     r'1      r'2      r'3
        # Values:        v'0     v'1      v'2      v'3
        # Lambda-values:         l'1      l'2      l'3
        # Entropies:    [e'0]   [e'1]    [e'2]
        # Continues:     c0      c'1      c'2      c'3
        # where z0 comes from the posterior, while z'i is the imagined states (prior)

        # Imagine trajectories in the latent space
        for i in range(1, self.horizon + 1):
            imagined_prior, recurrent_state = self.net.world_model.rssm.imagination(
                imagined_prior, recurrent_state, actions
            )
            imagined_prior = imagined_prior.view(1, -1, self.net.stochastic_state_size)
            imagined_latent_state = torch.cat((imagined_prior, recurrent_state), -1)
            imagined_trajectories[i] = imagined_latent_state
            imagined_actions[i] = actions = self.net.actor(imagined_latent_state.detach()).rsample()

        # Predict values, rewards and continues
        predicted_values_dist = itd.TwoHotEncodingDistribution(self.net.critic(imagined_trajectories.detach()), dims=1)
        predicted_target_values_dist = itd.TwoHotEncodingDistribution(
            self.net.target_critic(imagined_trajectories.detach()), dims=1
        )
        predicted_rewards = itd.TwoHotEncodingDistribution(
            self.net.world_model.reward_model(imagined_trajectories.detach()), dims=1
        ).mean
        predicted_continues = torch.exp(
            td.Independent(
                itd.BernoulliSafe(logits=self.net.world_model.continue_model(imagined_trajectories.detach())), 1
            ).log_prob(torch.ones_like(predicted_rewards))
        )

        # Denormalize the values
        value_offset, value_scale = self.net.value_normalizer.stats()
        predicted_values = predicted_values_dist.mean * value_scale + value_offset

        # Set up lambda returns
        discount = 1 if self.continuous_discounting else self.gamma
        weight = torch.cumprod(predicted_continues * discount, dim=0) / discount

        # Estimate lambda-values
        lambda_returns = compute_lambda_returns(
            predicted_rewards[1:],
            predicted_values[1:],
            predicted_continues[1:].unsqueeze(-1) * discount,
            lmbda=self.lmbda,
        )
        # Normalize lambda returns and compute advantages
        return_offset, return_scale = self.net.return_normalizer(lambda_returns, self.fabric)
        baseline = predicted_values[:-1]
        advantage = (lambda_returns - baseline) / return_scale
        advantage_offset, advantage_scale = self.net.advantage_normalizer(advantage, self.fabric)
        advantage_norm = (advantage - advantage_offset) / advantage_scale
        advantage_norm = advantage_norm.squeeze(-1)

        # Compute policy gradient loss with entropy regularization
        predicted_actions_dist = self.net.actor(imagined_trajectories.detach())
        log_pi = predicted_actions_dist.log_prob(imagined_actions.detach())[:-1]
        entropy = predicted_actions_dist.entropy()[:-1]
        policy_loss = -(weight[:-1].detach() * log_pi * advantage_norm.detach() + self.ent_coef * entropy).mean()

        # Normalize lambda returns
        value_offset, value_scale = self.net.value_normalizer(lambda_returns, self.fabric)
        lambda_returns_norm = (lambda_returns - value_offset) / value_scale
        lambda_returns_norm_padded = torch.cat([lambda_returns_norm, torch.zeros_like(lambda_returns_norm[:1])], dim=0)

        # Compute value loss
        value_loss = predicted_values_dist.log_prob(lambda_returns_norm_padded.detach())
        value_loss += predicted_values_dist.log_prob(predicted_target_values_dist.mean.detach())
        value_loss = -(weight[:-1].detach() * value_loss[:-1]).mean()

        return policy_loss, value_loss, entropy[:-1].mean()

    def forward(
        self,
        batch: TensorDict,
        dreamer_state: Tuple[Tensor, Tensor, Tensor],
        greedy: bool = False,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        """Predict actions.

        Args:
            batch (TensorDict): Batch with observation.
            dreamer_state (Tuple[Tensor, Tensor, Tensor]): Dreamer state.
            greedy (bool): Whether to use greedy actions.

        Returns:
            Tuple[Tensor, Tuple[Tensor, Tensor, Tensor]]: Actions and dreamer state.
        """
        return self.net(batch, dreamer_state, greedy)

    def configure_ratio(self) -> Ratio:
        """
        Configure ratio for training frequency based on environment steps.

        The replay ratio determines how much training (gradient updates) is performed
        per environment interaction step. This method converts the replay ratio into
        the actual number of gradient steps per environment step by accounting for
        the batch size in terms of time steps.

        Formula:
            gradient_steps_per_env_step = replay_ratio / (batch_size * block_size)

        Where:
        - replay_ratio: How many replay steps to use for training per env step
        - batch_size * block_size: Number of time steps consumed per gradient update

        Returns:
            Ratio: A ratio calculator that determines training repeats based on env steps
        """
        batch_steps = self.batch_size * self.block_size
        return Ratio(self.replay_ratio / batch_steps)

    def configure_replay_buffer(
        self,
        world_size: int,
        global_rank: int,
        checkpoint_dir: str,
    ) -> ReplayBuffer:
        buffer_size = int(self.replay_buffer_size // world_size)
        replay_buffer = ReplayBuffer(
            length=self.block_size,
            capacity=buffer_size,
            directory=os.path.join(checkpoint_dir, f"replay_buffer_{global_rank}"),
            chunksize=1024,
            online=True,
        )
        return replay_buffer

    def visualize_world_model(self, batch: TensorDict, max_batch_size: int = 6) -> Tensor:
        """
        Generate open-loop predictions for visualization.

        This function takes a batch with image observations and outputs a video tensor
        that shows the world model's ability to predict future observations. It uses the first half
        of each sequence for conditioning and predicts the second half.

        Strategy: Use first half of sequence for conditioning, predict second half
        - First half: Process with real observations (encoder + RSSM observe)
        - Second half: Generate predictions using RSSM imagination + decoder

        Args:
            batch (TensorDict): Batch from replay_buffer.sample_tensor() with shape [T, B, ...]
            max_batch_size (int): Maximum batch size for visualization

        Returns:
            Tensor: Video tensor with shape [T, C, 3H, B*W] showing true vs predicted observations
        """
        assert len(self.net.cnn_keys) >= 1, "World model needs at least one CNN key for visualization."

        # Get batch dimensions
        T, B = batch["is_first"].shape[:2]
        device = batch.device

        # Limit batch size for computational efficiency (similar to reference implementation)
        RB = min(max_batch_size, B)

        # Helper functions to split sequences in half
        firsthalf = lambda x: x[: T // 2, :RB] if len(x.shape) >= 2 else x
        secondhalf = lambda x: x[T // 2 :, :RB] if len(x.shape) >= 2 else x

        # Initialize recurrent and stochastic states
        recurrent_state = torch.zeros(1, RB, self.net.recurrent_state_size, device=device)
        posterior = torch.zeros(1, RB, self.net.stochastic_size, self.net.discrete_size, device=device)

        # ===== CONDITIONING PHASE (First Half) =====
        # Process first half with real observations to condition the world model

        # Prepare first half data
        first_obs = TensorDict({k: firsthalf(batch[k]) for k in self.net.obs_keys}, batch_size=(T // 2, RB))
        first_actions = firsthalf(batch["actions"])
        first_is_first = firsthalf(batch["is_first"])

        # Encode observations for first half
        embedded_obs_first = self.net.world_model.encoder(first_obs)

        # Process first half through RSSM with real observations
        obsfeat_list = []
        for t in range(T // 2):
            recurrent_state, posterior, _, _, _ = self.net.world_model.rssm.dynamic(
                posterior,
                recurrent_state,
                first_actions[t : t + 1],
                embedded_obs_first[t : t + 1],
                first_is_first[t : t + 1],
            )
            # Store latent features (concatenate stochastic and recurrent states)
            latent_state = torch.cat(
                (posterior.view(*posterior.shape[:-2], -1), recurrent_state), -1  # Flatten stochastic state
            )
            obsfeat_list.append(latent_state)

        obsfeat = torch.cat(obsfeat_list, dim=0)  # Shape: [T//2, RB, latent_size]

        # ===== PREDICTION PHASE (Second Half) =====
        # Imagine second half using learned dynamics (no real observations)

        # Prepare second half actions
        second_actions = secondhalf(batch["actions"])

        # Imagine trajectories for second half
        imgfeat_list = []
        imagined_prior = posterior.view(1, RB, -1)  # Flatten stochastic state for imagination

        for t in range(T - T // 2):
            # Use RSSM imagination to predict next state
            imagined_prior, recurrent_state = self.net.world_model.rssm.imagination(
                imagined_prior, recurrent_state, second_actions[t : t + 1]
            )
            # Store imagined latent features
            imagined_prior = imagined_prior.view(1, -1, self.net.stochastic_state_size)
            imagined_latent_state = torch.cat((imagined_prior, recurrent_state), -1)
            imgfeat_list.append(imagined_latent_state)

        imgfeat = torch.cat(imgfeat_list, dim=0)  # Shape: [T-T//2, RB, latent_size]

        # ===== RECONSTRUCTION =====
        # Decode both real and imagined features to observations

        # Reconstruct observations from real features (first half)
        obsrecons = self.net.world_model.decoder(obsfeat)

        # Reconstruct observations from imagined features (second half)
        imgrecons = self.net.world_model.decoder(imgfeat)

        # ===== VIDEO GENERATION =====
        # Create visualization comparing true vs predicted observations

        # Focus on the first CNN key for visualization (typically 'image' or 'rgb')
        if not self.net.cnn_keys:
            raise ValueError("No CNN keys available for visualization. World model needs image observations.")

        vis_key = self.net.cnn_keys[0]  # Use first image observation

        # Get true observations (limited to report batch size)
        true_obs = batch[vis_key][:, :RB]  # Shape: [T, RB, C, H, W]

        # Combine reconstructed (first half) and imagined (second half) predictions
        pred_first = obsrecons[vis_key]  # Shape: [T//2, RB, C, H, W]
        pred_second = imgrecons[vis_key]  # Shape: [T-T//2, RB, C, H, W]
        pred_obs = torch.cat([pred_first, pred_second], dim=0)  # Shape: [T, RB, C, H, W]

        # Undo image normalization
        if self.debug_transform:
            true_obs = self.debug_transform(true_obs)
            pred_obs = self.debug_transform(pred_obs)

        # Ensure proper dtype
        true_obs = true_obs.clamp(0, 255).to(torch.uint8)
        pred_obs = pred_obs.clamp(0, 255).to(torch.uint8)

        # Compute prediction error visualization
        error = ((pred_obs.float() - true_obs.float())).clamp(0, 255).to(torch.uint8)

        # Stack vertically: [true, predicted, error]
        video = torch.cat([true_obs, pred_obs, error], dim=3)  # Stack along height dimension [T, RB, C, 3H, W]

        # If grayscale, convert to RGB
        if video.shape[2] == 1:
            video = video.repeat(1, 1, 3, 1, 1)

        # Add borders to distinguish conditioning vs prediction phases
        # Add padding for borders
        video = F.pad(video, [2, 2, 2, 2, 0, 0, 0, 0, 0, 0])

        # Create border mask (interior should remain unchanged)
        mask = torch.zeros_like(video, dtype=torch.bool)
        mask[:, :, :, 2:-2, 2:-2] = True  # Interior mask

        # Create colored borders: green for conditioning, red for prediction
        green_border = torch.tensor([0, 255, 0], dtype=torch.uint8, device=device)
        red_border = torch.tensor([255, 0, 0], dtype=torch.uint8, device=device)

        # Reshape to image dimensions
        green_border = green_border.view(1, 1, 3, 1, 1).expand_as(video)
        red_border = red_border.view(1, 1, 3, 1, 1).expand_as(video)

        # Apply borders
        for t in range(T):
            # Use green for first half, red for second half
            color = green_border if t < T // 2 else red_border
            video[t] = torch.where(mask[t], video[t], color[t])

        # Convert to final format: [T, RB, C, 3H, W] -> [T, C, 3H, RB*W]
        video = video.permute(0, 2, 3, 1, 4)
        video = video.reshape(*video.shape[:-2], -1)

        return video
