import os

import wandb


class WandbModelCheckpoint:
    """Model checkpoint callback for Weights and Biases."""

    def save_model(self, **kwargs) -> None:
        """Save model to disk."""
        run = kwargs["trainer"].fabric.logger._experiment
        assert isinstance(run, wandb.sdk.wandb_run.Run), "Weights and Biases run not found."
        run.log_model(path=kwargs["path"], name=os.path.basename(kwargs["path"]))
