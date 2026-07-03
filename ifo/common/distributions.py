from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import (
    Bernoulli,
    Categorical,
    Distribution,
    Normal,
    OneHotCategoricalStraightThrough,
    constraints,
)
from torch.types import _size

from ifo.common.utils.functions import symexp, symlog


class TruncatedNormal(Normal):
    """
    Truncated Normal distribution with differentiable sampling.

    This class extends the standard Normal distribution to support truncation within
    specified bounds. It uses a straight-through estimator to maintain differentiability
    during sampling while enforcing hard constraints on the sampled values.

    The truncation is applied using a clamping operation with a straight-through gradient,
    allowing gradients to flow through the sampling operation while ensuring samples
    remain within the specified bounds.

    Args:
        loc (Tensor): Mean of the normal distribution.
        scale (Tensor): Standard deviation of the normal distribution.
        low (Tensor): Lower bound for truncation.
        high (Tensor): Upper bound for truncation.
        eps (float, optional): Small epsilon value to prevent samples from being exactly
            at the boundaries. Defaults to 1e-6.
        validate_args (bool, optional): Whether to validate input arguments. Defaults to None.
    """

    def __init__(
        self,
        loc: Tensor,
        scale: Tensor,
        low: Tensor,
        high: Tensor,
        eps: float = 1e-6,
        validate_args: Optional[bool] = None,
    ) -> None:
        """
        Initialize the TruncatedNormal distribution.

        Args:
            loc (Tensor): Mean of the normal distribution.
            scale (Tensor): Standard deviation of the normal distribution.
            low (Tensor): Lower bound for truncation.
            high (Tensor): Upper bound for truncation.
            eps (float, optional): Small epsilon value to prevent samples from being exactly
                at the boundaries. Defaults to 1e-6.
            validate_args (bool, optional): Whether to validate input arguments. Defaults to None.
        """
        super().__init__(loc, scale, validate_args)
        self.low = low
        self.high = high
        self.eps = eps

    def rsample(self, sample_shape: _size = torch.Size()) -> Tensor:
        """
        Generate reparameterized samples from the truncated normal distribution.

        This method samples from the underlying normal distribution and applies
        truncation using a straight-through estimator, which allows gradients to
        flow through the operation.

        Args:
            sample_shape (_size, optional): The shape of the samples to generate.
                Defaults to torch.Size().

        Returns:
            Tensor: Samples from the truncated normal distribution with the same
                dtype and device as the distribution parameters.
        """
        event = super().rsample(sample_shape)

        event_clamp = torch.clamp(event, self.low + self.eps, self.high - self.eps)

        event = event - event.detach() + event_clamp.detach()

        return event

    def sample(self, sample_shape: _size = torch.Size()) -> Tensor:
        """
        Generate samples from the truncated normal distribution without gradients.

        This is a non-differentiable version of rsample that disables gradient tracking.

        Args:
            sample_shape (_size, optional): The shape of the samples to generate.
                Defaults to torch.Size().

        Returns:
            Tensor: Samples from the truncated normal distribution without gradient information.
        """
        with torch.no_grad():
            return self.rsample(sample_shape)


class TwoHotEncodingDistribution(Distribution):
    """
    Two-hot encoding distribution for continuous values.

    This distribution represents continuous values using a categorical distribution over bins,
    where each value is encoded using two adjacent bins (two-hot encoding). This approach
    provides a more expressive representation than single-bin encoding while maintaining
    differentiability.

    The bins are constructed symmetrically around zero using symlog/symexp transformations
    to handle a wide range of values efficiently.

    Args:
        logits (Tensor): Unnormalized log probabilities for each bin.
        dims (int): Number of dimensions to treat as event dimensions. Defaults to 0.
        low (int): Lower bound for bin construction in symlog space. Defaults to -20.
        high (int): Upper bound for bin construction in symlog space. Defaults to 20.
        transfwd (Callable[[Tensor], Tensor]): Forward transformation function (e.g., symlog).
            Defaults to symlog.
        transbwd (Callable[[Tensor], Tensor]): Backward transformation function (e.g., symexp).
            Defaults to symexp.
        validate_args (bool, optional): Whether to validate input arguments. Defaults to None.
    """

    arg_constraints = {"logits": constraints.real}  # type: ignore[assignment]
    support = constraints.real  # type: ignore[assignment]
    has_rsample = False

    def __init__(
        self,
        logits: Tensor,
        dims: int = 0,
        low: int = -20,
        high: int = 20,
        transfwd: Callable[[Tensor], Tensor] = symlog,
        transbwd: Callable[[Tensor], Tensor] = symexp,
        validate_args: Optional[bool] = None,
    ):
        self.logits = logits
        self._probs = F.softmax(logits, dim=-1)
        self.dims = tuple([-x for x in range(1, dims + 1)])
        self.low = low
        self.high = high
        self.transfwd = transfwd
        self.transbwd = transbwd

        # Compute batch and event shapes
        batch_shape = logits.shape[: len(logits.shape) - dims - 1]
        event_shape = logits.shape[len(logits.shape) - dims - 1 : -1] + (1,)

        # Create bins using the same strategy as JAX implementation
        self.bins = self._create_bins(logits.shape[-1], logits.device)

        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def _create_bins(self, num_bins: int, device: torch.device) -> Tensor:
        """
        Create bins using symmetric construction with symexp transformation.

        Constructs bins symmetrically around zero to ensure numerical stability
        and balanced representation of positive and negative values.

        Args:
            num_bins (int): Number of bins to create.
            device (torch.device): Device on which to create the bins.

        Returns:
            Tensor: Bin centers with shape (num_bins,).
        """
        # Symmetric bin construction
        if num_bins % 2 == 1:
            # Odd number of bins
            half_size = (num_bins - 1) // 2 + 1
            half = torch.linspace(-20, 0, half_size, device=device, dtype=torch.float32)
            half = self.transbwd(half)  # Apply symexp
            # Create symmetric bins: [symexp(-20..0), -symexp(0..-20)[:-1]]
            bins = torch.cat([half, -half[:-1].flip(0)], dim=0)
        else:
            # Even number of bins
            half_size = num_bins // 2
            half = torch.linspace(-20, 0, half_size, device=device, dtype=torch.float32)
            half = self.transbwd(half)  # Apply symexp
            # Create symmetric bins: [symexp(-20..0), -symexp(0..-20)]
            bins = torch.cat([half, -half.flip(0)], dim=0)

        return bins

    @property
    def probs(self) -> Tensor:
        """
        Get the normalized probabilities for each bin.

        Returns:
            Tensor: Probability distribution over bins.
        """
        return self._probs

    @property
    def mean(self) -> Tensor:
        """
        Compute the mean of the distribution.

        Returns:
            Tensor: Mean value computed using symmetric prediction.
        """
        return self._symmetric_prediction()

    @property
    def mode(self) -> Tensor:
        """
        Compute the mode of the distribution.

        For this distribution, the mode is the same as the mean.

        Returns:
            Tensor: Mode value (same as mean).
        """
        # Mode is the same as mean for this distribution
        return self.mean

    def _symmetric_prediction(self) -> Tensor:
        """
        JAX-style symmetric prediction to avoid numerical errors.

        Uses symmetric computation when bins are symmetric around zero to improve
        numerical stability and reduce floating-point errors.

        Returns:
            Tensor: Predicted value as weighted average of bin centers.
        """
        n = self.logits.shape[-1]

        if n % 2 == 1:
            # Odd number of bins - split into three parts
            m = (n - 1) // 2
            p1 = self.probs[..., :m]  # Left half (negative)
            p2 = self.probs[..., m : m + 1]  # Center bin (around zero)
            p3 = self.probs[..., m + 1 :]  # Right half (positive)

            b1 = self.bins[..., :m]
            b2 = self.bins[..., m : m + 1]
            b3 = self.bins[..., m + 1 :]

            # Symmetric computation: center + (reversed_left + right)
            center_contrib = (p2 * b2).sum(dim=-1, keepdim=True)
            symmetric_contrib = ((p1 * b1).flip(-1) + (p3 * b3)).sum(dim=-1, keepdim=True)
            wavg = center_contrib + symmetric_contrib
        else:
            # Even number of bins - split into two parts
            half = n // 2
            p1 = self.probs[..., :half]  # Left half (negative)
            p2 = self.probs[..., half:]  # Right half (positive)

            b1 = self.bins[..., :half]
            b2 = self.bins[..., half:]

            # Symmetric computation: (reversed_left + right)
            wavg = ((p1 * b1).flip(-1) + (p2 * b2)).sum(dim=-1, keepdim=True)

        return wavg  # No need for transbwd since bins are already in target space

    def log_prob(self, value: Tensor) -> Tensor:
        """
        Compute the log probability of a value under this distribution.

        Uses two-hot encoding to compute the log probability by finding the two
        adjacent bins and computing their weighted contribution.

        Args:
            value (Tensor): Value(s) for which to compute log probability.

        Returns:
            Tensor: Log probability of the input value(s).
        """
        # Reshape bins for broadcasting comparison
        # bins: (num_bins,) -> add dimensions to broadcast with value
        # We need bins shape to be compatible: (..., 1, num_bins) where ... matches value[:-1]
        bins_shape = [1] * value.ndim
        bins_shape[-1] = len(self.bins)
        bins_expanded = self.bins.view(bins_shape)

        # Find adjacent bins (same logic as JAX implementation)
        # below in [-1, len(self.bins) - 1]
        below = (bins_expanded <= value).type(torch.int32).sum(dim=-1, keepdim=True) - 1
        # above in [0, len(self.bins)]
        above = len(self.bins) - (bins_expanded > value).type(torch.int32).sum(dim=-1, keepdim=True)

        # Clip to valid range
        above = torch.clip(above, 0, len(self.bins) - 1)
        below = torch.clip(below, 0, len(self.bins) - 1)

        # Handle case where target exactly matches a bin
        equal = below == above

        # Index bins using below/above indices
        bins_below = self.bins[below.squeeze(-1)].unsqueeze(-1)
        bins_above = self.bins[above.squeeze(-1)].unsqueeze(-1)

        # Compute distances and weights
        dist_to_below = torch.where(equal, torch.ones_like(equal, dtype=torch.float32), torch.abs(bins_below - value))
        dist_to_above = torch.where(equal, torch.ones_like(equal, dtype=torch.float32), torch.abs(bins_above - value))

        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total

        # Create target distribution using two-hot encoding
        target = (
            F.one_hot(below.squeeze(-1), len(self.bins)).float() * weight_below
            + F.one_hot(above.squeeze(-1), len(self.bins)).float() * weight_above
        )

        # Compute log probabilities
        log_pred = self.logits - torch.logsumexp(self.logits, dim=-1, keepdim=True)

        return (target * log_pred).sum(dim=self.dims)

    def sample(self, sample_shape: _size = torch.Size()) -> Tensor:
        """
        Sample from the distribution.

        Samples categorical indices according to the probability distribution,
        then converts them to continuous values using bin centers.

        Args:
            sample_shape (torch.Size): Shape of samples to generate. Defaults to torch.Size().

        Returns:
            Tensor: Sampled values.
        """
        # Convert sample_shape to torch.Size if necessary
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)

        # Sample categorical indices
        num_samples = sample_shape.numel() if sample_shape else 1
        categorical_samples = torch.multinomial(
            self.probs.reshape(-1, self.probs.shape[-1]), num_samples=num_samples, replacement=True
        )

        if sample_shape:
            categorical_samples = categorical_samples.view(*sample_shape, *self.probs.shape[:-1])
        else:
            categorical_samples = categorical_samples.view(*self.probs.shape[:-1])

        # Convert to continuous values using bin centers
        samples = self.bins[categorical_samples]

        # Add trailing dimension to match event_shape
        # event_shape should be (..., 1), so we need to add the 1 dimension
        samples = samples.unsqueeze(-1)

        return samples

    def entropy(self) -> Tensor:
        """
        Compute entropy of the distribution.

        Returns:
            Tensor: Entropy value(s).
        """
        log_probs = torch.log(self.probs + 1e-8)  # Add small epsilon for numerical stability
        return -(self.probs * log_probs).sum(dim=-1)

    @property
    def variance(self) -> Tensor:
        """
        Compute the variance of the distribution.

        Returns:
            Tensor: Variance of the distribution.
        """
        mean = self.mean
        # Compute E[X^2] - E[X]^2
        squared_bins = self.bins**2
        expected_x_squared = (self.probs * squared_bins).sum(dim=-1, keepdim=True)
        return expected_x_squared - mean**2

    def expand(self, batch_shape, _instance=None):
        """
        Returns a new distribution instance with batch dimensions expanded to `batch_shape`.

        Args:
            batch_shape: The desired expanded size.
            _instance: New instance provided by subclasses that need to override `.expand`.

        Returns:
            New distribution instance with batch dimensions expanded to batch_shape.
        """
        new = self._get_checked_instance(TwoHotEncodingDistribution, _instance)
        batch_shape = torch.Size(batch_shape)

        # Expand logits to new batch shape
        # Current logits shape: batch_shape + event_shape[:-1] + (num_bins,)
        # New logits shape: new_batch_shape + event_shape[:-1] + (num_bins,)
        num_bins = self.logits.shape[-1]
        param_shape = tuple(batch_shape) + tuple(self._event_shape[:-1]) + (num_bins,)
        new.logits = self.logits.expand(param_shape)
        new._probs = self._probs.expand(param_shape)

        # Copy other attributes
        new.dims = self.dims
        new.low = self.low
        new.high = self.high
        new.transfwd = self.transfwd
        new.transbwd = self.transbwd
        new.bins = self.bins  # bins are shared across batch

        # Initialize parent class
        super(TwoHotEncodingDistribution, new).__init__(batch_shape, self._event_shape, validate_args=False)
        new._validate_args = self._validate_args

        return new


class BernoulliSafe(Bernoulli):
    """
    A numerically stable Bernoulli distribution.

    This class extends the standard Bernoulli distribution with numerically stable
    log probability computation using logsigmoid instead of log(sigmoid).
    """

    def __init__(
        self,
        probs: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        validate_args: Optional[bool] = None,
    ) -> None:
        """
        Initialize the BernoulliSafe distribution.

        Args:
            probs: Event probabilities. Should be in the range [0, 1].
            logits: Event log-odds (log(p/(1-p))). Unbounded real values.
            validate_args: Whether to validate input arguments.
        """
        super().__init__(probs, logits, validate_args)

    @property
    def mode(self) -> torch.Tensor:
        """
        Returns the mode of the distribution.

        Returns:
            The mode (most likely value) of the distribution, which is 1 if p > 0.5, else 0.
        """
        mode = (self.probs > 0.5).to(self.probs)
        return mode

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Calculate the log probability of the given value.

        Uses numerically stable logsigmoid computation instead of log(sigmoid).

        Args:
            value: The value for which to compute the log probability.

        Returns:
            The log probability of the value.
        """
        logp = torch.nn.functional.logsigmoid(self.logits)
        lognotp = torch.nn.functional.logsigmoid(-self.logits)
        return value * logp + (1 - value) * lognotp


class MultiCategorical(Distribution):
    """Multi-categorical distribution for multi-discrete action spaces.

    This distribution is useful for environments with multiple independent categorical
    actions (e.g., MultiDiscrete action spaces in Gymnasium). Each categorical distribution
    can have a different number of categories.

    The distribution maintains a list of independent Categorical distributions and provides
    unified methods for sampling, computing log probabilities, and computing entropy.

    Args:
        probs (List[Tensor], optional): Event probabilities for each categorical distribution.
            Each tensor should have shape (batch_size, num_categories_i) where num_categories_i
            is the number of categories for the i-th categorical distribution.
        logits (List[Tensor], optional): Event log probabilities (unnormalized) for each
            categorical distribution. Same shape requirements as probs.
        validate_args (bool, optional): Whether to validate the arguments. Defaults to None.

    Raises:
        ValueError: If neither probs nor logits are provided.
        ValueError: If distributions have inconsistent batch shapes.

    Examples:
        >>> # Create a multi-categorical distribution for 3 actions with 2, 3, and 4 categories
        >>> logits = [torch.randn(10, 2), torch.randn(10, 3), torch.randn(10, 4)]
        >>> dist = MultiCategorical(logits=logits)
        >>> actions = dist.sample()  # Returns list of 3 tensors with shape (10,)
        >>> log_probs = dist.log_prob(actions)  # Returns tensor with shape (10,)
    """

    arg_constraints = {}  # type: ignore[assignment]
    has_rsample = False

    def __init__(
        self,
        probs: Optional[List[Tensor]] = None,
        logits: Optional[List[Tensor]] = None,
        validate_args: Optional[bool] = None,
    ) -> None:
        """Initializes the multi-categorical distribution.

        Args:
            probs (List[Tensor], optional): Event probabilities for each categorical
                distribution. Defaults to None.
            logits (List[Tensor], optional): Event log probabilities for each categorical
                distribution. Defaults to None.
            validate_args (bool, optional): Whether to validate the arguments.
                Defaults to None.
        Raises:
            ValueError: If neither probs nor logits are provided.
            ValueError: If distributions have inconsistent batch shapes.
        """
        if probs is None and logits is None:
            raise ValueError("Either probs or logits must be provided.")

        if probs is not None:
            if not isinstance(probs, list):
                raise ValueError("probs must be a list of tensors.")
            self.distributions = [Categorical(probs=p, validate_args=validate_args) for p in probs]
        else:
            if not isinstance(logits, list):
                raise ValueError("logits must be a list of tensors.")
            self.distributions = [Categorical(logits=logit, validate_args=validate_args) for logit in logits]

        # Enforce that all distributions have the same batch shape to calculate valid log_probs.
        batch_shapes = [dist.batch_shape for dist in self.distributions]
        if not all([batch_shape == batch_shapes[0] for batch_shape in batch_shapes]):
            raise ValueError(f"All distributions must have the same batch shape. " f"Got shapes: {batch_shapes}")

        # Initialize parent class with proper batch_shape
        super().__init__(batch_shape=batch_shapes[0], validate_args=validate_args)

    @property
    def logits(self) -> List[Tensor]:
        """Returns the logits for each categorical distribution.

        Returns:
            List[Tensor]: List of logit tensors, one per categorical distribution.
        """
        return [dist.logits for dist in self.distributions]

    @property
    def probs(self) -> List[Tensor]:
        """Returns the probabilities for each categorical distribution.

        Returns:
            List[Tensor]: List of probability tensors, one per categorical distribution.
        """
        return [dist.probs for dist in self.distributions]

    @property
    def mean(self) -> List[Tensor]:  # type: ignore[override]
        """Returns the mean of each categorical distribution.

        For categorical distributions, the mean is the expected category index.

        Returns:
            List[Tensor]: List of mean tensors with shape (batch_size,), one per distribution.
        """
        return [dist.mean for dist in self.distributions]

    @property
    def mode(self) -> List[Tensor]:  # type: ignore[override]
        """Returns the mode (most likely category) of each categorical distribution.

        Returns:
            List[Tensor]: List of mode tensors with shape (batch_size,), one per distribution.
        """
        return [dist.mode for dist in self.distributions]

    @property
    def variance(self) -> Tensor:
        """Returns the average variance across all categorical distributions.

        Note: This computes the mean of variances across distributions, providing a
        scalar metric of overall uncertainty. This is not the true joint variance.

        Returns:
            Tensor: Average variance across all distributions.
        """
        variances = [dist.variance for dist in self.distributions]
        return torch.stack(variances).mean(dim=0)

    def sample(self, sample_shape: _size = torch.Size()) -> List[Tensor]:  # type: ignore[override]
        """Sample from each categorical distribution independently.

        Args:
            sample_shape (_size, optional): The shape of samples to generate.
                Defaults to torch.Size().

        Returns:
            List[Tensor]: List of sampled category indices, one tensor per distribution.
                Each tensor has shape sample_shape + batch_shape.
        """
        return [dist.sample(sample_shape) for dist in self.distributions]

    def log_prob(self, value: List[Tensor]) -> Tensor:  # type: ignore[override]
        """Compute the log probability of the given values.

        The total log probability is the sum of log probabilities from each
        categorical distribution, assuming independence.

        Args:
            value (List[Tensor]): List of category indices, one tensor per distribution.
                Each tensor should have shape matching the batch_shape.

        Returns:
            Tensor: Sum of log probabilities with shape batch_shape.

        Raises:
            ValueError: If the number of values doesn't match the number of distributions.
        """
        if len(value) != len(self.distributions):
            raise ValueError(f"Expected {len(self.distributions)} values but got {len(value)}")
        log_probs = [dist.log_prob(val) for val, dist in zip(value, self.distributions)]
        return torch.stack(log_probs).sum(dim=0)

    def entropy(self) -> Tensor:
        """Compute the total entropy across all categorical distributions.

        The total entropy is the sum of entropies from each distribution,
        assuming independence.

        Returns:
            Tensor: Sum of entropies with shape batch_shape.
        """
        return torch.stack([dist.entropy() for dist in self.distributions]).sum(dim=0)


class MultiOneHotCategoricalStraightThrough(Distribution):
    """Multi one-hot straight-through categorical distribution for multi-discrete actions.

    Mirrors MultiCategorical but uses OneHotCategoricalStraightThrough for each head, returning
    one-hot actions for differentiable sampling.

    Args:
        probs (List[Tensor], optional): Probabilities for each categorical head.
        logits (List[Tensor], optional): Logits for each categorical head.
        validate_args (bool, optional): Whether to validate the arguments.
    """

    arg_constraints = {}  # type: ignore[assignment]
    has_rsample = True

    def __init__(
        self,
        probs: Optional[List[Tensor]] = None,
        logits: Optional[List[Tensor]] = None,
        validate_args: Optional[bool] = None,
    ) -> None:
        if probs is None and logits is None:
            raise ValueError("Either probs or logits must be provided.")

        if probs is not None:
            if not isinstance(probs, list):
                raise ValueError("probs must be a list of tensors.")
            self.distributions = [OneHotCategoricalStraightThrough(probs=p, validate_args=validate_args) for p in probs]
        else:
            if not isinstance(logits, list):
                raise ValueError("logits must be a list of tensors.")
            self.distributions = [
                OneHotCategoricalStraightThrough(logits=logit, validate_args=validate_args) for logit in logits
            ]

        batch_shapes = [dist.batch_shape for dist in self.distributions]
        if not all([batch_shape == batch_shapes[0] for batch_shape in batch_shapes]):
            raise ValueError(f"All distributions must have the same batch shape. Got shapes: {batch_shapes}")

        super().__init__(batch_shape=batch_shapes[0], validate_args=validate_args)

    @property
    def logits(self) -> List[Tensor]:
        return [dist.logits for dist in self.distributions]

    @property
    def probs(self) -> List[Tensor]:
        return [dist.probs for dist in self.distributions]

    @property
    def mean(self) -> List[Tensor]:  # type: ignore[override]
        # For one-hot categorical, mean equals probabilities (expected one-hot vector)
        return [dist.probs for dist in self.distributions]

    @property
    def mode(self) -> List[Tensor]:  # type: ignore[override]
        return [dist.mode for dist in self.distributions]

    @property
    def variance(self) -> Tensor:
        # Average variance across heads
        variances = [dist.variance for dist in self.distributions]
        return torch.stack(variances).mean(dim=0)

    def sample(self, sample_shape: _size = torch.Size()) -> List[Tensor]:  # type: ignore[override]
        return [dist.sample(sample_shape) for dist in self.distributions]

    def rsample(self, sample_shape: _size = torch.Size()) -> List[Tensor]:  # type: ignore[override]
        return [dist.rsample(sample_shape) for dist in self.distributions]

    def log_prob(self, value: List[Tensor]) -> Tensor:  # type: ignore[override]
        if len(value) != len(self.distributions):
            raise ValueError(f"Expected {len(self.distributions)} values but got {len(value)}")
        log_probs = [dist.log_prob(val) for val, dist in zip(value, self.distributions)]
        return torch.stack(log_probs).sum(dim=0)

    def entropy(self) -> Tensor:
        return torch.stack([dist.entropy() for dist in self.distributions]).sum(dim=0)


class DictDistribution(Distribution):
    """Dict distribution for dictionary-structured action spaces.

    This class combines multiple distributions into a single distribution by treating them
    as a dictionary. This is useful for environments with Dict action spaces in Gymnasium,
    where each action component can have a different distribution type.

    The class automatically handles rsample support if any sub-distribution supports it,
    and provides unified methods for sampling, computing log probabilities, and computing entropy.

    Args:
        distributions (Dict[str, Distribution]): Dictionary of distributions where keys are
            distribution names and values are torch.distributions.Distribution instances.

    Raises:
        ValueError: If distributions dictionary is empty.
        ValueError: If distributions have inconsistent batch shapes.

    Examples:
        >>> # Create a dict distribution for a robot with continuous position and discrete mode
        >>> from torch.distributions import Normal, Categorical, Independent
        >>> distributions = {
        ...     'position': Independent(Normal(torch.zeros(10, 2), torch.ones(10, 2)), 1),
        ...     'mode': Categorical(logits=torch.randn(10, 3))
        ... }
        >>> dist = DictDistribution(distributions)
        >>> actions = dist.sample()  # Returns {'position': Tensor, 'mode': Tensor}
        >>> log_probs = dist.log_prob(actions)  # Returns tensor with shape (10,)
    """

    arg_constraints = {}  # type: ignore[assignment]
    has_rsample = False

    def __init__(self, distributions: Dict[str, Distribution]) -> None:
        """Initializes the DictDistribution.

        Args:
            distributions (Dict[str, Distribution]): Dictionary of distributions where keys are
                distribution names and values are torch.distributions.Distribution instances.

        Raises:
            ValueError: If distributions dictionary is empty.
            ValueError: If distributions have inconsistent batch shapes.
        """
        if not distributions:
            raise ValueError("distributions dictionary cannot be empty")

        self.has_rsample = any([getattr(v, "has_rsample", False) for v in distributions.values()])
        self.distributions = distributions

        # Enforce that all distributions have the same batch shape to calculate valid log_probs.
        batch_shapes = [dist.batch_shape for dist in self.distributions.values()]
        if not all([batch_shape == batch_shapes[0] for batch_shape in batch_shapes]):
            raise ValueError(
                f"All distributions must have the same batch shape. "
                f"Got: {dict(zip(distributions.keys(), batch_shapes))}"
            )
        super().__init__(batch_shape=batch_shapes[0])

    @property
    def mean(self) -> Dict[str, Tensor]:  # type: ignore[override]
        """Returns the mean of each distribution as a dictionary.

        Returns:
            Dict[str, Tensor]: Dictionary mapping distribution names to mean tensors.
        """
        return {k: v.mean for k, v in self.distributions.items()}

    @property
    def mode(self) -> Dict[str, Tensor]:  # type: ignore[override]
        """Returns the mode of each distribution as a dictionary.

        Returns:
            Dict[str, Tensor]: Dictionary mapping distribution names to mode tensors.
        """
        return {k: v.mode for k, v in self.distributions.items()}

    @property
    def variance(self) -> Tensor:
        """Returns the average variance across all distributions.

        Note: This computes the mean of variances across distributions. For nested
        DictDistributions, this may not represent the true joint variance but provides
        a useful scalar metric.

        Returns:
            Tensor: Average variance across all sub-distributions.
        """
        variances = [v.variance for v in self.distributions.values()]
        return torch.stack([v.mean() if v.numel() > 1 else v for v in variances]).mean()

    def sample(self, sample_shape: _size = torch.Size()) -> Dict[str, Tensor]:  # type: ignore[override]
        """Sample from each distribution.

        Args:
            sample_shape (_size, optional): The shape of the sample to draw.
                Defaults to torch.Size().

        Returns:
            Dict[str, Tensor]: Dictionary mapping distribution names to sampled tensors.
        """
        return {k: v.sample(sample_shape) for k, v in self.distributions.items()}

    def rsample(self, sample_shape: _size = torch.Size()) -> Dict[str, Tensor]:  # type: ignore[override]
        """Reparameterized sample from each distribution (if supported).

        Falls back to regular sampling for distributions that don't support rsample.

        Args:
            sample_shape (_size, optional): The shape of the sample to draw.
                Defaults to torch.Size().

        Returns:
            Dict[str, Tensor]: Dictionary mapping distribution names to sampled tensors.
        """
        return {
            k: v.rsample(sample_shape) if getattr(v, "has_rsample", False) else v.sample(sample_shape)
            for k, v in self.distributions.items()
        }

    def log_prob(self, value: Dict[str, Tensor]) -> Tensor:  # type: ignore[override]
        """Compute the log probability of the given value.

        The total log probability is the sum of log probabilities from each sub-distribution,
        assuming independence between the distributions.

        Args:
            value (Dict[str, Tensor]): Dictionary mapping distribution names to action tensors.

        Returns:
            Tensor: Sum of log probabilities across all distributions with shape batch_shape.

        Raises:
            ValueError: If value keys don't match distribution keys.
        """
        if set(value.keys()) != set(self.distributions.keys()):
            raise ValueError(
                f"Value keys {set(value.keys())} don't match " f"distribution keys {set(self.distributions.keys())}"
            )
        log_probs = [self.distributions[k].log_prob(v) for k, v in value.items()]
        return torch.stack(log_probs).sum(dim=0)

    def entropy(self) -> Tensor:
        """Compute the total entropy.

        The total entropy is the sum of entropies from each sub-distribution,
        assuming independence between the distributions.

        Returns:
            Tensor: Sum of entropies across all distributions with shape batch_shape.
        """
        return torch.stack([v.entropy() for v in self.distributions.values()]).sum(dim=0)


class SymlogDistribution:
    """
    Distribution for continuous values in symlog space.

    This distribution models continuous values by transforming them to symlog space
    and computing distances (MSE or absolute error) between predictions and targets.
    The symlog transformation allows the model to handle values across a wide range
    of magnitudes while maintaining numerical stability.

    Args:
        mode (Tensor): The predicted mode in symlog space.
        dims (int): Number of trailing dimensions to treat as event dimensions.
        dist (str): Distance metric to use. Either "mse" for mean squared error or
            "abs" for absolute error. Defaults to "mse".
        agg (str): Aggregation method for the distance. Either "sum" or "mean".
            Defaults to "sum".
        tol (float): Tolerance threshold below which distances are set to zero.
            Defaults to 1e-8.

    Attributes:
        mode (Tensor): The predicted mode transformed back to original space via symexp.
        mean (Tensor): The predicted mean (same as mode for this distribution).
    """

    def __init__(
        self,
        mode: Tensor,
        dims: int,
        dist: str = "mse",
        agg: str = "sum",
        tol: float = 1e-8,
    ):
        self._mode = mode
        self._dims = tuple([-x for x in range(1, dims + 1)])
        self._dist = dist
        self._agg = agg
        self._tol = tol
        self._batch_shape = mode.shape[: len(mode.shape) - dims]
        self._event_shape = mode.shape[len(mode.shape) - dims :]

    @property
    def mode(self) -> Tensor:
        """Return the mode of the distribution in original space."""
        return symexp(self._mode)

    @property
    def mean(self) -> Tensor:
        """Return the mean of the distribution in original space."""
        return symexp(self._mode)

    def log_prob(self, value: Tensor) -> Tensor:
        """
        Compute the log probability (negative loss) of the given value.

        Args:
            value (Tensor): The target value in original space.

        Returns:
            Tensor: Negative distance between prediction and target, aggregated
                over event dimensions.
        """
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
        if self._dist == "mse":
            distance = (self._mode - symlog(value)) ** 2
            distance = torch.where(distance < self._tol, 0, distance)
        elif self._dist == "abs":
            distance = torch.abs(self._mode - symlog(value))
            distance = torch.where(distance < self._tol, 0, distance)
        else:
            raise NotImplementedError(self._dist)
        if self._agg == "mean":
            loss = distance.mean(self._dims)
        elif self._agg == "sum":
            loss = distance.sum(self._dims)
        else:
            raise NotImplementedError(self._agg)
        return -loss


class MSEDistribution(Distribution):
    """
    Distribution for continuous values using mean squared error.

    This distribution models continuous values by computing the mean squared error
    between predictions and targets. Unlike SymlogDistribution, this operates
    directly in the original space without transformation.

    Args:
        mode (Tensor): The predicted mode.
        dims (int): Number of trailing dimensions to treat as event dimensions.
        agg (str): Aggregation method for the distance. Either "sum" or "mean".
            Defaults to "sum".

    Attributes:
        mode (Tensor): The predicted mode.
        mean (Tensor): The predicted mean (same as mode for this distribution).
    """

    arg_constraints = {"mode": constraints.real}  # type: ignore[assignment]
    support = constraints.real  # type: ignore[assignment]
    has_rsample = False

    def __init__(
        self,
        mode: Tensor,
        dims: int,
        agg: str = "sum",
        validate_args: Optional[bool] = None,
    ):
        self._mode = mode
        self._dims = tuple([-x for x in range(1, dims + 1)])
        self._agg = agg
        batch_shape = mode.shape[: len(mode.shape) - dims]
        event_shape = mode.shape[len(mode.shape) - dims :]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    @property
    def mode(self) -> Tensor:
        """Return the mode of the distribution."""
        return self._mode

    @property
    def mean(self) -> Tensor:
        """Return the mean of the distribution."""
        return self._mode

    def log_prob(self, value: Tensor) -> Tensor:
        """
        Compute the log probability (negative MSE loss) of the given value.

        Args:
            value (Tensor): The target value.

        Returns:
            Tensor: Negative mean squared error between prediction and target,
                aggregated over event dimensions.
        """
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
        distance = (self._mode - value) ** 2
        if self._agg == "mean":
            loss = distance.mean(self._dims)
        elif self._agg == "sum":
            loss = distance.sum(self._dims)
        else:
            raise NotImplementedError(self._agg)
        return -loss
