import os
from typing import List, Optional, Union

import torch
from tensordict import TensorDict
from torch import Tensor
from torch.utils.data import Dataset


class ReplayBuffer:
    """A circular replay buffer based on TensorDict."""

    REQUIRED_KEYS = {"observation", "action", "reward", "terminated", "truncated"}

    def __init__(
        self,
        capacity: int,
        num_envs: int = 1,
        block_size: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        is_frame_stacked: bool = False,
        padding_type: str = "zero",
    ) -> None:
        """Initialize the replay buffer.

        Args:
            capacity (int): Maximum number of transitions to store in the buffer.
            num_envs (int): Number of parallel environments.
            block_size (int, optional): Length of sequences to sample from the buffer. Sequences will
                be padded with zeros to the left to ensure that the last observation of each sequence
                is the same. If None, sequences will not be sampled.
            device (Union[str, torch.device]): Device to store the buffer on. Can be either a string
                (e.g., "cpu", "cuda") or a torch.device object. Defaults to "cpu".
            is_frame_stacked (bool): Whether the observations of the transitions are frame stacked using
                gymnasium.wrappers.FrameStack. This affects how observations are handled during
                sampling and storage and can lead to errors if not correctly specified. Defaults to False.
            padding_type (str): Type of padding to use. At the moment only "zero" is supported.
        """
        assert (
            not is_frame_stacked or block_size is not None
        ), "Block size must be specified if frame stacking is enabled."
        assert padding_type == "zero", "Only zero padding is supported at the moment."
        self.usable_capacity = capacity
        self.full_capacity = capacity + (block_size - 1) if block_size else capacity
        self.num_envs = num_envs
        self.block_size = block_size
        self.device = torch.device(device)
        self._is_frame_stacked = is_frame_stacked
        self._padding_type = padding_type
        self.reset()

    def _validate_transition(self, transition: TensorDict) -> None:
        """Validate that a transition contains all required keys and has the correct shape.

        Args:
            transition (TensorDict): The transition to validate

        Raises:
            ValueError: If any required keys are missing
        """
        assert isinstance(transition, TensorDict), "Transition must be of type TensorDict."
        transition_keys = set(transition.keys())
        missing_keys = self.REQUIRED_KEYS - transition_keys
        if missing_keys:
            raise ValueError(f"Transition is missing required keys: {missing_keys}")
        assert (
            transition.batch_size[0] == self.num_envs
        ), f"Transition must have {self.num_envs} environments, but has {transition.batch_size[0]}."

    def reset(self) -> None:
        """Reset the buffer."""
        # Buffer to store transitions
        self._buffer = TensorDict({}, batch_size=[self.num_envs, self.full_capacity], device=self.device)
        self._mask = None
        self._padding_lengths = None
        if self.block_size:
            # Mask to indicate which transitions are valid
            self._mask = torch.zeros(
                self.num_envs, self.full_capacity, self.block_size, dtype=torch.bool, device=self.device
            )
            # Number of padding steps into the past for each transition.
            self._padding_lengths = torch.full(
                (self.num_envs,), self.block_size - 1, dtype=torch.int32, device=self.device
            )
        # Current index in the buffer
        self._current_idx = 0
        # Start index in the buffer
        self._start_idx = 0
        # Number of transitions in the buffer
        self.length = 0

    def _add_batched(self, transitions: TensorDict) -> None:
        """Add a batch of transitions to the buffer efficiently.

        Args:
            transitions (TensorDict): A TensorDict with shape [num_envs, time_steps, ...].
                Must contain keys: observation, action, reward, terminated, truncated

        Raises:
            ValueError: If any required keys are missing or shapes don't match
            ValueError: If the number of transitions exceeds the capacity
        """
        if transitions.batch_size[1] > self.usable_capacity:
            raise ValueError(
                f"Number of transitions ({transitions.batch_size[1]}) exceeds buffer capacity ({self.usable_capacity})"
            )

        # Get the number of time steps
        time_steps = transitions.batch_size[1]

        # Store transitions in the buffer
        # Calculate indices for the buffer
        start_idx = self._current_idx
        remaining_capacity = self.full_capacity - start_idx
        remaining_capacity_steps = min(remaining_capacity, time_steps)
        overhang_capacity_steps = time_steps - remaining_capacity_steps

        # Fill the remaining capacity of the buffer
        if remaining_capacity_steps > 0:
            self._buffer[:, start_idx : start_idx + remaining_capacity_steps] = transitions[
                :, :remaining_capacity_steps
            ]

        # Fill the overhang capacity of the buffer
        if overhang_capacity_steps > 0:
            self._buffer[:, :overhang_capacity_steps] = transitions[:, remaining_capacity_steps:]

        # Update start position
        if self.length == self.usable_capacity:
            self._start_idx = (self._start_idx + time_steps) % self.full_capacity
        elif self.length + time_steps > self.usable_capacity:
            self._start_idx = (self._start_idx + overhang_capacity_steps) % self.full_capacity

        # Update size
        self.length = min(self.length + time_steps, self.usable_capacity)

        # Update sample mask and position
        self._current_idx = (self._current_idx + time_steps) % self.full_capacity

    def append(self, transition: TensorDict) -> None:
        """Add a single transition to the buffer.

        Args:
            transition (TensorDict): A TensorDict containing the transition data.
                Must contain keys: observation, action, reward, terminated, truncated
                If num_envs > 1, the tensors should have a batch dimension of size num_envs

        Raises:
            ValueError: If any required keys are missing
        """
        self._validate_transition(transition)

        # Bring transition to the correct device
        transition = transition.to(self.device)

        # Take only the last observation
        if self._is_frame_stacked:
            transition["observation"] = transition["observation"][:, -1]

        # Add transition at current position
        idx = (self._start_idx + self.length) % self.full_capacity
        self._buffer[:, idx] = transition

        # Add mask and update padding lengths
        if self.block_size:
            mask = torch.arange(self.block_size, device=self.device)[None, :] >= self._padding_lengths[:, None]
            self._mask[:, idx] = mask
            self._padding_lengths = torch.maximum(torch.zeros_like(self._padding_lengths), self._padding_lengths - 1)
            done = transition["terminated"] | transition["truncated"]
            self._padding_lengths[done] = self.block_size - 1

        # Update pointers and size
        if self.length < self.usable_capacity:
            self.length += 1
        else:
            self._start_idx = (self._start_idx + 1) % self.full_capacity

    def extend(self, transitions: TensorDict) -> None:
        """Extend the buffer with multiple transitions.

        Args:
            transitions (TensorDict): A TensorDict where the second dimension is time (shape: [num_envs, time_steps, ...]).
                Must contain keys: observation, action, reward, terminated, truncated

        Raises:
            ValueError: If any required keys are missing in any transition
            ValueError: If the number of transitions exceeds the capacity
            ValueError: If the first dimension doesn't match num_envs
        """
        # Check shape
        if transitions.batch_size[0] != self.num_envs:
            raise ValueError(
                f"Expected first dimension of transitions to be {self.num_envs} (num_envs), "
                f"got {transitions.batch_size[0]}"
            )

        # Validate all required keys are present
        self._validate_transition(transitions)
        for i in range(transitions.batch_size[1]):
            self.append(transitions[:, i])

    def __len__(self) -> int:
        """Get the current number of transitions in the buffer.

        Returns:
            int: Number of transitions currently in the buffer
        """
        return self.length

    def _resolve_indices(self, idx):
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self.length)
            indices = (torch.arange(start, stop, step, device=self.device) + self._start_idx) % self.full_capacity
            return indices.tolist()
        elif isinstance(idx, int):
            if idx < 0:
                idx += self.length
            if not 0 <= idx < self.length:
                raise IndexError("Index out of range")
            return (self._start_idx + idx) % self.full_capacity
        else:
            raise TypeError("Invalid index type")

    def __getitem__(self, idx):
        if isinstance(idx, str):
            # rb["a"] → full view of key "a"
            return ReplayBufferKeyView(self, idx)[:]
        elif isinstance(idx, slice) or isinstance(idx, int):
            indices = self._resolve_indices(idx)
            return self._buffer[:, indices]

    def __setitem__(self, idx, value):
        if isinstance(idx, str):
            # rb["a"] = value → full view of key "a"
            ReplayBufferKeyView(self, idx)[:] = value
        elif isinstance(idx, slice) or isinstance(idx, int):
            indices = self._resolve_indices(idx)
            if isinstance(indices, list):
                if len(value) != len(indices):
                    raise ValueError("Length mismatch in assignment")
                self._buffer[:, indices] = value
            else:
                self._buffer[:, indices] = value
        else:
            raise TypeError("Invalid index type")

    def sample(self, batch_size: int) -> TensorDict:
        """Sample randomly a batch of transitions from the buffer with replacement.

        Args:
            batch_size (int): Number of transition sequences to sample

        Returns:
            TensorDict: A TensorDict containing the sampled transitions with shape [batch_size, block_size, ...]
            if block_size > 1, otherwise [batch_size, ...]
        """
        if self.length * self.num_envs < batch_size:
            raise ValueError(
                f"Not enough samples in the buffer with block_size {self.block_size} "
                f"({self.length * self.num_envs} samples available, but {batch_size} requested)"
            )
        # Sample random env indices and time indices starting from start_idx
        random_env_indices = torch.randint(self.num_envs, (batch_size,), device=self.device)
        random_time_indices = self._start_idx + torch.randint(self.length, (batch_size,), device=self.device)
        random_time_indices = random_time_indices % self.full_capacity

        if self.block_size:
            # Get slicing indices with block_size for env and time indices
            slice_random_env_indices = random_env_indices.view(-1, 1).expand(-1, self.block_size)
            slice_random_time_indices = random_time_indices[:, None] + torch.arange(
                1 - self.block_size, 1, device=self.device
            )

            # Get the samples and pad with zeros if we have time steps that are not available
            samples = self._buffer[slice_random_env_indices, slice_random_time_indices]
            mask = self._mask[random_env_indices, random_time_indices]
            samples[~mask] = 0
        else:
            samples = self._buffer[random_env_indices, random_time_indices]

        return samples

    def save(self, path: str) -> None:
        """Save the buffer to disk.

        Args:
            path (str): Path to save the buffer to
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "_buffer": self._buffer,
                "_mask": self._mask,
                "_padding_lengths": self._padding_lengths,
                "_start_idx": self._start_idx,
                "length": self.length,
                "usable_capacity": self.usable_capacity,
                "full_capacity": self.full_capacity,
                "num_envs": self.num_envs,
                "block_size": self.block_size,
                "device": self.device,
                "_is_frame_stacked": self._is_frame_stacked,
                "_padding_type": self._padding_type,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load the buffer from disk.

        Args:
            path (str): Path to load the buffer from
        """
        checkpoint = torch.load(path)
        self._buffer = checkpoint["_buffer"]
        self._mask = checkpoint["_mask"]
        self._padding_lengths = checkpoint["_padding_lengths"]
        self._start_idx = checkpoint["_start_idx"]
        self.length = checkpoint["length"]
        self.usable_capacity = checkpoint["usable_capacity"]
        self.full_capacity = checkpoint["full_capacity"]
        self.num_envs = checkpoint["num_envs"]
        self.block_size = checkpoint["block_size"]
        self.device = checkpoint["device"]
        self._is_frame_stacked = checkpoint["_is_frame_stacked"]
        self._padding_type = checkpoint["_padding_type"]

    @property
    def dataset(self) -> Dataset:
        """Get a PyTorch dataset for the buffer.

        Returns:
            Dataset: A PyTorch dataset containing all transitions in the buffer
        """
        return ReplayBufferDataset(self)

    def keys(self):
        """Get the keys of the buffer.

        Returns:
            A generator of the keys of the buffer.
        """
        return self._buffer.keys()


class ReplayBufferDataset(Dataset[TensorDict]):
    """A PyTorch Dataset that provides access to transitions in a ReplayBuffer.

    This dataset allows sampling sequences of transitions from the replay buffer
    for training. It handles the circular nature of the buffer and ensures proper
    indexing of sequences.
    """

    def __init__(self, replay_buffer: ReplayBuffer) -> None:
        """Initialize the dataset.

        Args:
            replay_buffer (ReplayBuffer): The replay buffer to create a dataset from

        Raises:
            ValueError: If the buffer is empty
        """
        if replay_buffer._buffer is None or replay_buffer.length == 0:
            raise ValueError("Buffer is empty")
        self._buffer = replay_buffer._buffer
        self._mask = replay_buffer._mask
        self._padding_lengths = replay_buffer._padding_lengths
        self.length = replay_buffer.length * replay_buffer.num_envs
        self._num_envs = replay_buffer.num_envs
        self._size = replay_buffer.length
        self.device = replay_buffer.device
        self.block_size = replay_buffer.block_size
        self._start_idx = replay_buffer._start_idx
        self._full_capacity = replay_buffer.full_capacity

    def __len__(self) -> int:
        """Get the number of valid samples in the dataset.

        Returns:
            int: Number of valid samples that can be sampled from the buffer
        """
        return self.length

    def __getitem__(self, idx: int) -> TensorDict:
        """Get a single sequence of transitions.

        Args:
            idx: Index of the sequence to retrieve

        Returns:
            TensorDict: A TensorDict containing a sample
        """
        # Get environment and time indices
        env_idx = idx // self._size
        time_idx = idx % self._size
        time_idx = (time_idx + self._start_idx) % self._full_capacity

        if self.block_size:
            # Get slicing indices with block_size for env and time indices
            slice_env_idx = torch.as_tensor([env_idx], device=self.device).expand(self.block_size)
            slice_time_idx = torch.as_tensor([time_idx], device=self.device) + torch.arange(
                1 - self.block_size, 1, device=self.device
            )

            # Get the samples and pad with zeros if we have time steps that are not available
            sample = self._buffer[slice_env_idx, slice_time_idx]
            mask = self._mask[env_idx, time_idx]
            sample[~mask] = 0
        else:
            sample = self._buffer[env_idx, time_idx]

        return sample

    def __getitems__(self, indices: List[int]) -> TensorDict:
        """Get multiple sequences of transitions efficiently.

        Args:
            indices (List[int]): List of indices of sequences to retrieve

        Returns:
            TensorDict: A TensorDict containing multiple sequences of transitions, each of length block_size
        """
        indices = torch.as_tensor(indices, device=self.device)
        env_indices = indices // self._size
        time_indices = indices % self._size
        time_indices = (time_indices + self._start_idx) % self._full_capacity

        if self.block_size:
            # Get slicing indices with block_size for env and time indices
            slice_env_indices = env_indices.view(-1, 1).expand(-1, self.block_size)
            slice_time_indices = time_indices[:, None] + torch.arange(1 - self.block_size, 1, device=self.device)

            # Get the samples and pad with zeros if we have time steps that are not available
            samples = self._buffer[slice_env_indices, slice_time_indices]
            mask = self._mask[env_indices, time_indices]
            samples[~mask] = 0
        else:
            samples = self._buffer[env_indices, time_indices]

        return samples


class ReplayBufferKeyView:
    """A view into a specific key of a ReplayBuffer that supports indexing operations.

    This class provides a convenient way to access and modify values for a specific key
    in the replay buffer using standard Python indexing syntax and automatically handles
    the circular nature of the buffer.
    """

    def __init__(self, replay_buffer: ReplayBuffer, key: str) -> None:
        """Initialize the key view.

        Args:
            replay_buffer (ReplayBuffer): The replay buffer to create a key view for
            key (str): The key to view in the buffer
        """
        self._replay_buffer = replay_buffer
        self._key = key

    def __getitem__(self, idx: Union[int, slice]) -> Tensor:
        """Get values for the key at the specified indices.

        Args:
            idx (Union[int, slice]): Integer or slice index to retrieve values for

        Returns:
            Tensor: A Tensor containing the values at the specified indices

        Raises:
            TypeError: If idx is not an integer or slice
        """
        if isinstance(idx, slice) or isinstance(idx, int):
            indices = self._replay_buffer._resolve_indices(idx)
            return self._replay_buffer._buffer[self._key][:, indices]
        else:
            raise TypeError("Invalid index type")

    def __setitem__(self, idx: Union[int, slice], value: Tensor) -> None:
        """Set values for the key at the specified indices.

        Args:
            idx (Union[int, slice]): Integer or slice index to set values for
            value (Tensor): Tensor of values to set

        Raises:
            TypeError: If idx is not an integer or slice
            ValueError: If length of value doesn't match number of indices when using list indices
        """
        if isinstance(idx, slice) or isinstance(idx, int):
            indices = self._replay_buffer._resolve_indices(idx)
            if isinstance(indices, list):
                if len(value) != len(indices):
                    raise ValueError("Length mismatch in assignment")
                self._replay_buffer._buffer[self._key][:, indices] = value
            else:
                self._replay_buffer._buffer[self._key][:, indices] = value
        else:
            raise TypeError("Invalid index type")

    def __len__(self) -> int:
        """Get the number of elements in the view.

        Returns:
            int: Number of elements in the replay buffer
        """
        return len(self._replay_buffer)

    def __repr__(self):
        return repr(self[:])
