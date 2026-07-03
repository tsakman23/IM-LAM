from tensordict import TensorDict

from ifo.common.trainer import SupervisedTrainer


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
