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
