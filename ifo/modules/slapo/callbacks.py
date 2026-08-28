import torch
from tensordict import TensorDict

from ifo.common.nets.mask_biased_attention import MaskBiasedAttention
from ifo.common.trainer import SupervisedTrainer


class ExtractionBetaAnneal:
    """Linearly anneal the (frozen) extraction mask-bias ``beta`` from ``start`` to ``end`` over
    ``anneal_steps`` training steps, then hold at ``end``.

    Train under the known-good soft bias early, tighten toward the hard-gate late, so the hard 
    constraint is never imposed before the model has learned useful features. Operates only on 
    ``MaskBiasedAttention`` ``beta`` BUFFERS - i.e. modules built with a frozen ``extraction_beta`` 
    (set ``net.world_model.extraction_beta`` to the ``start`` value); learnable-``beta`` modules 
    are left untouched. The annealed value is visible in W&B via the existing
    ``beta_msa_a``/``beta_msa_o`` diagnostics, since they read the same buffer.
    """

    def __init__(self, start: float, end: float, anneal_steps: int) -> None:
        """Initialize the callback.

        Args:
            start (float): Beta at step 0 (should match ``net.world_model.extraction_beta``).
            end (float): Beta held after ``anneal_steps`` (the hard-gate strength).
            anneal_steps (int): Steps over which beta ramps linearly from ``start`` to ``end``.
        """
        self.start, self.end, self.anneal_steps = float(start), float(end), int(anneal_steps)

    def train_batch_start(self, trainer: SupervisedTrainer, model, **kwargs) -> None:
        """Set every frozen extraction beta to the annealed value for the current step."""
        frac = min(1.0, trainer.global_step / max(1, self.anneal_steps))
        value = self.start + (self.end - self.start) * frac
        for module in model.modules():
            beta = getattr(module, "beta", None)
            if isinstance(module, MaskBiasedAttention) and beta is not None and not beta.requires_grad:
                beta.data.fill_(value)


class ObjectLossWeightAnneal:
    """Warm up the dual loss's object term: hold ``object_loss_weight`` at ``start`` for ``hold_steps``,
    then ramp it linearly to ``end`` over ``anneal_steps``, then hold at ``end``.

    Applying the full ``lambda_o`` from step 0 lets the dual loss's amplified small-object gradient hit
    the latent-injection pathway before the agent branch and the latent ``z`` have settled, which drives
    a latent-scale runaway in IM-LAM (the action decoder's NMSE explodes). Warming the object term up -
    typically from ``start=0`` (object branch dormant) - lets the agent pathway stabilize first, closer
    to the stable union-loss regime, before the object budget is introduced.

    Operates only on the module's ``object_loss_weight`` BUFFER (a scalar tensor); modules that store it
    as a plain float, or not at all, are left untouched. Mutating the buffer in place is torch.compile
    safe - the value is a graph input, not a guard - so no recompile is triggered as it ramps.
    """

    def __init__(self, start: float, end: float, anneal_steps: int, hold_steps: int = 0) -> None:
        """Initialize the callback.

        Args:
            start (float): Weight during the hold window and at the start of the ramp (e.g. ``0.0``).
            end (float): Weight held after the ramp (the target ``lambda_o``, e.g. ``1.0``).
            anneal_steps (int): Steps over which the weight ramps linearly from ``start`` to ``end``.
            hold_steps (int): Steps to hold at ``start`` before the ramp begins. Defaults to 0.
        """
        self.start, self.end = float(start), float(end)
        self.anneal_steps, self.hold_steps = int(anneal_steps), int(hold_steps)

    def train_batch_start(self, trainer: SupervisedTrainer, model, **kwargs) -> None:
        """Set the (buffered) object_loss_weight to the annealed value for the current step."""
        step = trainer.global_step
        if step < self.hold_steps:
            value = self.start
        else:
            frac = min(1.0, (step - self.hold_steps) / max(1, self.anneal_steps))
            value = self.start + (self.end - self.start) * frac
        weight = getattr(model, "object_loss_weight", None)
        if isinstance(weight, torch.Tensor):
            weight.fill_(value)


class StopThresholdIOU:
    """Stop training once mIoU exceeds a threshold and trigger a final validation."""

    def __init__(self, threshold: float) -> None:
        """Initialize the callback.

        Args:
            threshold (float): The threshold for the mIoU.
        """
        assert 0 <= threshold <= 1, "Threshold must be between 0 and 1."
        self.threshold = threshold

    def train_batch_end(self, trainer: SupervisedTrainer, step_dict: TensorDict, **kwargs) -> None:
        """Stop training once mIoU exceeds a threshold and trigger a final validation.

        Args:
            trainer (SupervisedTrainer): The trainer of the model.
            step_dict (TensorDict): The step dictionary containing the metrics.
        """
        assert "miou" in step_dict, "mIoU must be in the step dictionary."
        miou = step_dict["miou"].item()
        if miou >= self.threshold:
            trainer.max_steps = trainer.global_step
            trainer.validation_frequency = trainer.global_step
