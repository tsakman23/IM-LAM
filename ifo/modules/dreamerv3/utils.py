import contextlib
import io
import string
import sys
import threading
import traceback
import uuid as uuidlib
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Independent, OneHotCategoricalStraightThrough


class Ratio:
    """
    A ratio-based calculator for determining training repeats based on environment steps.

    This class implements a mechanism to calculate how many training iterations should be
    performed based on the current environment step and a specified ratio. It maintains
    internal state to track progress and ensure consistent scheduling.

    The implementation is adapted from Hafner et al. (2023) DreamerV3:
    https://github.com/danijar/dreamerv3/blob/8fa35f83eee1ce7e10f3dee0b766587d0a713a60/dreamerv3/embodied/core/when.py#L26

    Example:
        >>> ratio = Ratio(0.5)  # Train 0.5 times per environment step
        >>> ratio(10)  # Returns number of training repeats for step 10
        1
        >>> ratio(20)  # Returns number of training repeats for step 20
        5
    """

    def __init__(self, ratio: float):
        """
        Initialize the Ratio calculator.

        Args:
            ratio (float): The ratio for calculating repeats. Must be non-negative.

        Raises:
            ValueError: If ratio is negative.
        """
        assert ratio > 0, f"Ratio must be non-negative, got {ratio}"
        self._ratio = ratio
        self._prev = None

    def __call__(self, step: int) -> int:
        if self._ratio == 0:
            return 0
        if self._prev is None:
            self._prev = step
            return 1
        repeats = int((step - self._prev) * self._ratio)
        self._prev += repeats / self._ratio
        return repeats


def compute_stochastic_state(logits: Tensor, discrete: int = 32, sample=True) -> Tensor:
    """
    Compute the stochastic state from the logits computed by the transition or
    representation model.

    Args:
        logits (Tensor): logits from either the representation model or the transition model.
        discrete (int, optional): the size of the Categorical variables.
            Defaults to 32.
        sample (bool): whether or not to sample the stochastic state.
            Default to True.

    Returns:
        The sampled stochastic state.
    """
    logits = logits.view(*logits.shape[:-1], -1, discrete)
    dist = Independent(OneHotCategoricalStraightThrough(logits=logits), 1)
    stochastic_state = dist.rsample() if sample else dist.mode
    return stochastic_state


# Adapted from: https://github.com/NM512/dreamerv3-torch/blob/main/tools.py#L929
def init_weights(module: nn.Module) -> None:
    """
    Initialize weights for neural network modules using truncated normal distribution.

    This function implements DreamerV3's weight initialization scheme, which uses
    a truncated normal distribution with variance scaled by the fan-in and fan-out
    of the layer. The scaling factor 0.87962566103423978 is the inverse of the
    standard deviation of a truncated normal distribution.

    Args:
        module: The neural network module to initialize. Supports nn.Linear,
               nn.Conv2d, nn.ConvTranspose2d, and nn.LayerNorm layers.
    """
    if isinstance(module, nn.Linear):
        input_features = module.in_features
        output_features = module.out_features
        average_fan = (input_features + output_features) / 2.0
        variance_scale = 1.0 / average_fan
        # Inverse of std of truncated normal distribution
        truncated_normal_correction = 0.87962566103423978
        standard_deviation = np.sqrt(variance_scale) / truncated_normal_correction
        nn.init.trunc_normal_(
            module.weight.data,
            mean=0.0,
            std=standard_deviation,
            a=-2.0 * standard_deviation,
            b=2.0 * standard_deviation,
        )
        if module.bias is not None:
            module.bias.data.fill_(0.0)
    elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        kernel_spatial_size = module.kernel_size[0] * module.kernel_size[1]
        input_features = kernel_spatial_size * module.in_channels
        output_features = kernel_spatial_size * module.out_channels
        average_fan = (input_features + output_features) / 2.0
        variance_scale = 1.0 / average_fan
        # Inverse of std of truncated normal distribution
        truncated_normal_correction = 0.87962566103423978
        standard_deviation = np.sqrt(variance_scale) / truncated_normal_correction
        nn.init.trunc_normal_(module.weight.data, mean=0.0, std=standard_deviation, a=-2.0, b=2.0)
        if module.bias is not None:
            module.bias.data.fill_(0.0)
    elif isinstance(module, nn.LayerNorm):
        module.weight.data.fill_(1.0)
        if module.bias is not None:
            module.bias.data.fill_(0.0)


# Adapted from: https://github.com/NM512/dreamerv3-torch/blob/main/tools.py#L957
def uniform_init_weights(base_scale: float) -> Callable[[nn.Module], None]:
    """
    Create a weight initialization function that uses uniform distribution.

    This function returns a callable that initializes neural network weights
    using a uniform distribution. The variance is scaled by the provided base_scale
    and the fan-in/fan-out of each layer. The uniform distribution bounds are
    computed as sqrt(3 * variance) to match the variance of a normal distribution.

    Args:
        base_scale: Base scaling factor for weight initialization variance.
                   Higher values lead to larger initial weights.

    Returns:
        A callable function that can be used with nn.Module.apply() to initialize
        weights of Linear and LayerNorm layers.
    """

    def initialize_module_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            input_features = module.in_features
            output_features = module.out_features
            average_fan = (input_features + output_features) / 2.0
            variance_scale = base_scale / average_fan
            # For uniform distribution: var = (b-a)^2/12, so limit = sqrt(3*var) for same variance as normal
            uniform_limit = np.sqrt(3 * variance_scale)
            nn.init.uniform_(module.weight.data, a=-uniform_limit, b=uniform_limit)
            if hasattr(module.bias, "data"):
                module.bias.data.fill_(0.0)
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            if hasattr(module.bias, "data"):
                module.bias.data.fill_(0.0)

    return initialize_module_weights


def compute_lambda_returns(
    rewards: Tensor,
    values: Tensor,
    continues: Tensor,
    lmbda: float = 0.95,
) -> Tensor:
    """
    Compute lambda returns for temporal difference learning.

    The lambda return is computed recursively as:
    G_t^λ = r_t + γ * c_t * [(1-λ) * V(s_{t+1}) + λ * G_{t+1}^λ]

    Where:
    - r_t is the reward at time t
    - γ * c_t represents the discount factor (continues)
    - V(s_{t+1}) is the value function estimate
    - λ (lambda) controls the mixing between TD(0) and Monte Carlo returns

    Args:
        rewards (Tensor): Rewards sequence of shape [T, ...] where T is the time horizon.
        values (Tensor): Value function estimates of shape [T+1, ...]. The last value
                        V(s_T) is used as the bootstrap value for the final timestep.
        continues (Tensor): Continuation flags of shape [T, ...]. Typically γ * (1 - done)
                           where γ is the discount factor and done indicates episode termination.
        lmbda (float, optional): Lambda parameter controlling the mixing between TD(0) and
                               Monte Carlo returns. Values closer to 0 rely more on value
                               function estimates (lower variance, higher bias), while values
                               closer to 1 rely more on actual returns (higher variance, lower bias).
                               Defaults to 0.95.

    Returns:
        Tensor: Lambda returns of shape [T, ...], where each element G_t^λ represents
               the lambda return starting from timestep t.

    Note:
        This implementation processes the sequence backwards (from T-1 to 0) to efficiently
        compute the recursive lambda returns in a single pass.
    """
    assert (
        rewards.shape == values.shape == continues.shape
    ), f"Rewards, values, and continues must have the same shape, got {rewards.shape}, {values.shape}, {continues.shape}"
    vals = [values[-1:]]
    interm = rewards + continues * values * (1 - lmbda)
    for t in reversed(range(len(continues))):
        vals.append(interm[t] + continues[t] * lmbda * vals[-1])
    ret = torch.cat(list(reversed(vals))[:-1])
    return ret


def timestamp(time: Optional[datetime] = None, millis: bool = False) -> str:
    """
    Generate a timestamp string from a datetime object.

    Args:
      time: Datetime object to convert. If None, uses current time.
      millis: If True, includes microseconds in the timestamp.

    Returns:
      Timestamp string in format YYYYMMDDTHHMMSS or YYYYMMDDTHHMMSSFmicroseconds.
    """
    if time is None:
        time = datetime.now()
    string = time.strftime("%Y%m%dT%H%M%S")
    if millis:
        string += f"F{time.microsecond:06d}"
    return string


class UUID:
    """
    UUID that is stored as 16 byte string and can be converted to and from
    int, string, and array types.
    """

    __slots__ = ("value", "_hash")

    DEBUG_ID: Optional[int] = None
    BASE62 = string.digits + string.ascii_letters
    BASE62REV = {x: i for i, x in enumerate(BASE62)}

    @classmethod
    def reset(cls, *, debug: bool) -> None:
        """Reset UUID counter for debug mode."""
        cls.DEBUG_ID = 0 if debug else None

    def __init__(self, value: Optional[Union[int, str, bytes, np.ndarray, "UUID"]] = None) -> None:
        """
        Initialize a UUID.

        Args:
          value: Value to initialize from. Can be None (generates new UUID),
                 int, str, bytes, numpy array, or another UUID.
        """
        if value is None:
            if self.DEBUG_ID is None:
                self.value = uuidlib.uuid4().bytes
            else:
                debug_id = type(self).DEBUG_ID
                assert debug_id is not None
                debug_id += 1
                type(self).DEBUG_ID = debug_id
                self.value = debug_id.to_bytes(16, "big")
        elif isinstance(value, UUID):
            self.value = value.value
        elif isinstance(value, int):
            self.value = value.to_bytes(16, "big")
        elif isinstance(value, bytes):
            assert len(value) == 16, value
            self.value = value
        elif isinstance(value, str):
            if self.DEBUG_ID is None:
                integer = 0
                for index, char in enumerate(value[::-1]):
                    integer += (62**index) * self.BASE62REV[char]
                self.value = integer.to_bytes(16, "big")
            else:
                self.value = int(value).to_bytes(16, "big")
        elif isinstance(value, np.ndarray):
            self.value = value.tobytes()
        else:
            raise ValueError(value)
        assert type(self.value) == bytes, type(self.value)  # noqa
        assert len(self.value) == 16, len(self.value)
        self._hash = hash(self.value)

    def __int__(self) -> int:
        """Convert UUID to integer."""
        return int.from_bytes(self.value, "big")

    def __str__(self) -> str:
        """Convert UUID to base62 string representation."""
        if self.DEBUG_ID is not None:
            return str(int(self))
        chars = []
        integer = int(self)
        while integer != 0:
            chars.append(self.BASE62[integer % 62])
            integer //= 62
        while len(chars) < 22:
            chars.append("0")
        return "".join(chars[::-1])

    def __array__(self) -> np.ndarray:
        """Convert UUID to numpy array."""
        return np.frombuffer(self.value, np.uint8)

    def __getitem__(self, index: int) -> Any:
        """Index into UUID bytes."""
        return self.__array__()[index]

    def __repr__(self) -> str:
        """String representation of UUID."""
        return str(self)

    def __eq__(self, other: Any) -> bool:
        """Check equality with another UUID."""
        return self.value == other.value

    def __hash__(self) -> int:
        """Return hash of UUID."""
        return self._hash

    @property
    def uuid(self) -> "UUID":
        """Return self for compatibility."""
        return self


class Chunk:
    """
    A fixed-size container for storing sequential time-series data.

    Chunks store data in pre-allocated NumPy arrays and can be saved/loaded
    to/from compressed .npz files. Each chunk has a unique UUID and tracks
    its successor chunk for maintaining ordered sequences.

    Attributes:
      time: Timestamp when the chunk was created (in milliseconds).
      uuid: Unique identifier for this chunk.
      succ: UUID of the successor chunk (or 0 if no successor).
      length: Number of valid entries currently stored.
      size: Maximum capacity of the chunk.
      data: Dictionary mapping keys to NumPy arrays storing the actual data.
      saved: Whether this chunk has been saved to disk.
    """

    __slots__ = ("time", "uuid", "succ", "length", "size", "data", "saved")

    def __init__(self, size: int = 1024) -> None:
        """
        Initialize a new empty chunk.

        Args:
          size: Maximum number of steps this chunk can hold. Defaults to 1024.
        """
        self.time = timestamp(millis=True)
        self.uuid = UUID()
        self.succ = UUID(0)
        self.length = 0
        self.size = size
        self.data: Optional[Dict[str, np.ndarray]] = None
        self.saved = False

    def __repr__(self) -> str:
        """Return string representation of the chunk."""
        return f"Chunk({self.filename})"

    def __lt__(self, other: "Chunk") -> bool:
        """
        Compare chunks by timestamp for sorting.

        Args:
          other: Another Chunk instance to compare with.

        Returns:
          True if this chunk's timestamp is earlier than the other's.
        """
        return self.time < other.time

    @property
    def filename(self) -> str:
        """
        Generate the filename for this chunk when saved to disk.

        The filename format is: {timestamp}-{uuid}-{successor_uuid}-{length}.npz

        Returns:
          The filename string for this chunk.
        """
        succ = self.succ.uuid if isinstance(self.succ, type(self)) else self.succ
        return f"{self.time}-{str(self.uuid)}-{str(succ)}-{self.length}.npz"

    @property
    def nbytes(self) -> int:
        """
        Calculate the total memory size of the stored data in bytes.

        Returns:
          Total number of bytes occupied by all arrays in this chunk.
        """
        if not self.data:
            return 0
        return sum(x.nbytes for x in self.data.values())

    def append(self, step: Dict[str, np.ndarray]) -> None:
        """
        Append a single step of data to the chunk.

        On the first append, allocates storage arrays based on the shapes and dtypes
        of the provided data. Subsequent appends must have matching keys.

        Args:
          step: Dictionary mapping keys to NumPy arrays representing one timestep.

        Raises:
          AssertionError: If the chunk is already full.
        """
        assert self.length < self.size
        if not self.data:
            example = step
            self.data = {k: np.empty((self.size, *v.shape), v.dtype) for k, v in example.items()}
        for key, value in step.items():
            self.data[key][self.length] = value
        self.length += 1
        # if self.length == self.size:
        #   [x.setflags(write=False) for x in self.data.values()]

    def update(self, index: int, length: int, mapping: Dict[str, np.ndarray]) -> None:
        """
        Update a contiguous range of entries in the chunk.

        Args:
          index: Starting index for the update.
          length: Number of entries to update.
          mapping: Dictionary mapping keys to arrays with shape (length, ...).

        Raises:
          AssertionError: If the specified range is out of bounds.
        """
        assert 0 <= index <= self.length, (index, self.length)
        assert 0 <= index + length <= self.length, (index, length, self.length)
        assert self.data is not None, "Cannot update chunk with no data"
        for key, value in mapping.items():
            self.data[key][index : index + length] = value

    def slice(self, index: int, length: int) -> Dict[str, np.ndarray]:
        """
        Extract a contiguous slice of data from the chunk.

        Args:
          index: Starting index for the slice.
          length: Number of entries to extract.

        Returns:
          Dictionary mapping keys to sliced arrays with shape (length, ...).

        Raises:
          AssertionError: If the specified range is out of bounds.
        """
        assert 0 <= index and index + length <= self.length
        assert self.data is not None, "Cannot slice chunk with no data"
        return {k: v[index : index + length] for k, v in self.data.items()}

    def save(self, directory: Union[str, Path], log: bool = False) -> None:
        """
        Save the chunk to disk as a compressed .npz file.

        Only the valid portion of the data (up to self.length) is saved.
        After saving, the chunk is marked as saved and cannot be saved again.

        Args:
          directory: Directory path where the chunk file should be saved.
          log: If True, print a message after saving. Defaults to False.

        Raises:
          AssertionError: If the chunk has already been saved.
        """
        assert not self.saved
        assert self.data is not None, "Cannot save chunk with no data"
        self.saved = True
        filename = Path(directory) / self.filename
        data = {k: v[: self.length] for k, v in self.data.items()}
        with io.BytesIO() as stream:
            np.savez_compressed(stream, **data)
            stream.seek(0)
            filename.write_bytes(stream.read())
        if log:
            print(f"Saved chunk: {filename.name}")

    @classmethod
    def load(cls, filename: Union[str, Path], error: Literal["raise", "none"] = "raise") -> Optional["Chunk"]:
        """
        Load a chunk from a .npz file on disk.

        The filename must follow the format: {time}-{uuid}-{succ}-{length}.npz

        Args:
          filename: Path to the chunk file to load.
          error: Error handling mode. 'raise' will re-raise exceptions,
                 'none' will return None on error. Defaults to 'raise'.

        Returns:
          The loaded Chunk instance, or None if error='none' and loading failed.

        Raises:
          AssertionError: If error parameter is not 'raise' or 'none'.
          Exception: If loading fails and error='raise'.
        """
        assert error in ("raise", "none")
        path = Path(filename)
        time, uuid, succ, length = path.stem.split("-")
        length = int(length)
        try:
            with path.open("rb") as f:
                data = np.load(f)
                data = {k: data[k] for k in data.keys()}
        except Exception:
            tb = "".join(traceback.format_exception(*sys.exc_info()))
            print(f"Error loading chunk {filename}:\n{tb}")
            if error == "raise":
                raise
            else:
                return None
        chunk = cls(length)
        chunk.time = time
        chunk.uuid = UUID(uuid)
        chunk.succ = UUID(succ)
        chunk.length = length
        chunk.data = data
        chunk.saved = True
        return chunk


class RWLock:
    """
    A readers-writer lock implementation using threading primitives.

    Allows multiple concurrent readers or a single exclusive writer.
    Writers have priority over readers to prevent writer starvation.

    Usage:
      lock = RWLock()

      # For reading:
      with lock.reading:
          # Read operations
          pass

      # For writing:
      with lock.writing:
          # Write operations
          pass
    """

    def __init__(self) -> None:
        """Initialize the readers-writer lock."""
        self.lock = threading.Lock()
        self.active_writer_lock = threading.Lock()
        self.writer_count = 0
        self.waiting_reader_count = 0
        self.active_reader_count = 0
        self.readers_finished_cond = threading.Condition(self.lock)
        self.writers_finished_cond = threading.Condition(self.lock)

    @property
    @contextlib.contextmanager
    def reading(self):
        """
        Context manager for acquiring read access.

        Multiple readers can hold the lock simultaneously.
        Blocks if a writer is active or waiting.

        Yields:
          None: Control to the caller while holding read lock.
        """
        try:
            self.acquire_read()
            yield
        finally:
            self.release_read()

    @property
    @contextlib.contextmanager
    def writing(self):
        """
        Context manager for acquiring write access.

        Only one writer can hold the lock at a time.
        Blocks until all readers have finished.

        Yields:
          None: Control to the caller while holding write lock.
        """
        try:
            self.acquire_write()
            yield
        finally:
            self.release_write()

    def acquire_read(self) -> None:
        """
        Acquire read access to the lock.

        Blocks if writers are active or waiting. Multiple readers
        can acquire the lock simultaneously.
        """
        with self.lock:
            if self.writer_count:
                self.waiting_reader_count += 1
                while self.writer_count:
                    self.writers_finished_cond.wait()
                self.waiting_reader_count -= 1
            self.active_reader_count += 1

    def release_read(self) -> None:
        """
        Release read access to the lock.

        Notifies waiting writers if this was the last active reader.
        """
        with self.lock:
            assert self.active_reader_count > 0
            self.active_reader_count -= 1
            if not self.active_reader_count and self.writer_count:
                self.readers_finished_cond.notify_all()

    def acquire_write(self) -> None:
        """
        Acquire write access to the lock.

        Blocks until all active readers have finished. Only one
        writer can hold the lock at a time.
        """
        with self.lock:
            self.writer_count += 1
            while self.active_reader_count:
                self.readers_finished_cond.wait()
        self.active_writer_lock.acquire()

    def release_write(self) -> None:
        """
        Release write access to the lock.

        Notifies waiting readers if no more writers are waiting.
        """
        self.active_writer_lock.release()
        with self.lock:
            assert self.writer_count > 0
            self.writer_count -= 1
            if not self.writer_count and self.waiting_reader_count:
                self.writers_finished_cond.notify_all()


class Uniform:
    """
    A thread-safe uniform random sampler for replay buffer items.

    Maintains a collection of keys and provides uniform random sampling
    with O(1) insertion, deletion, and sampling operations.

    Attributes:
      indices: Dictionary mapping keys to their positions in the keys list.
      keys: List of all available keys for sampling.
      rng: NumPy random number generator for sampling.
      lock: Threading lock for thread-safe operations.
    """

    def __init__(self, seed: int = 0) -> None:
        """
        Initialize the uniform sampler.

        Args:
          seed: Random seed for reproducible sampling. Defaults to 0.
        """
        self.indices = {}
        self.keys = []
        self.rng = np.random.default_rng(seed)
        self.lock = threading.Lock()

    def __len__(self) -> int:
        """Return the number of keys available for sampling."""
        return len(self.keys)

    def __call__(self):
        """
        Sample a random key uniformly from the available keys.

        Returns:
          A randomly selected key from the collection.

        Raises:
          ValueError: If no keys are available for sampling.
        """
        with self.lock:
            if not self.keys:
                raise ValueError("Cannot sample from empty uniform sampler")
            # numpy 1.24+ returns Python int directly for low/high scalar
            index = int(self.rng.integers(0, len(self.keys)))
            return self.keys[index]

    def __setitem__(self, key, stepids) -> None:
        """
        Add a new key to the sampler.

        Args:
          key: The key to add for sampling.
          stepids: Associated step IDs (unused in uniform sampling but kept for interface compatibility).
        """
        with self.lock:
            if key not in self.indices:
                self.indices[key] = len(self.keys)
                self.keys.append(key)

    def __delitem__(self, key) -> None:
        """
        Remove a key from the sampler.

        Uses swap-and-pop for O(1) deletion by moving the last element
        to the position of the deleted element.

        Args:
          key: The key to remove from sampling.

        Raises:
          AssertionError: If trying to delete when fewer than 2 keys remain.
          KeyError: If the key is not found in the sampler.
        """
        with self.lock:
            assert 2 <= len(self), len(self)
            index = self.indices.pop(key)
            last = self.keys.pop()
            if index != len(self.keys):
                self.keys[index] = last
                self.indices[last] = index
