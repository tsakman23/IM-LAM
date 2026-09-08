from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.checkpoint import get_latest_checkpoint, load_checkpoint, save_checkpoint


class Trainer:
    """Generic supervised trainer with pure PyTorch + wandb logging.

    Supports both epoch-based and step-based training, gradient clipping,
    validation, checkpointing on best val loss, and wandb logging.
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        wandb_enabled: bool = True,
        precision: str = "bfloat16",
        device: str = "cuda",
        limit_train_batches: Optional[int] = None,
        limit_val_batches: Optional[int] = None,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.wandb_enabled = wandb_enabled
        self.precision = precision
        self.device = device
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.global_step = 0
        self.global_epoch = 0
        self.best_val_loss = float("inf")

        # Setup autocast dtype
        if precision == "bfloat16":
            self.autocast_dtype = torch.bfloat16
        elif precision == "float16":
            self.autocast_dtype = torch.float16
        else:
            self.autocast_dtype = torch.float32

    def fit(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        compute_loss: Callable,
        max_epochs: Optional[int] = None,
        max_steps: Optional[int] = None,
        grad_clip: float = 0.0,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        val_loader: Optional[DataLoader] = None,
        val_frequency: int = 1,
        val_unit: str = "epoch",
        log_frequency: int = 1,
        checkpoint_prefix: str = "model",
        extra_log_fn: Optional[Callable] = None,
        extra_val_fn: Optional[Callable[[nn.Module, float, int, int], None]] = None,
        torch_compile: bool = False,
    ) -> Dict[str, Any]:
        """Main training loop.

        Args:
            model: model to train.
            optimizer: optimizer.
            train_loader: training data loader.
            compute_loss: callable(model, batch) -> (loss, metrics_dict).
            max_epochs: max epochs (mutually exclusive with max_steps).
            max_steps: max gradient steps.
            grad_clip: max gradient norm (0 = no clipping).
            scheduler: optional LR scheduler (stepped per optimizer step).
            val_loader: optional validation loader.
            val_frequency: validate every N units (see val_unit).
            val_unit: unit for val_frequency, either "epoch" or "step".
            log_frequency: log metrics every N steps.
            checkpoint_prefix: filename prefix for checkpoints.
            extra_log_fn: optional callable(model, batch, step) for extra logging (e.g., images).
            extra_val_fn: optional callable(model, val_loss, global_step, global_epoch) called right after
                each validation run.
            torch_compile: compile model with torch.compile(mode='max-autotune').

        Returns:
            Dict with final training metrics.
        """
        model = model.to(self.device)
        scaler = torch.amp.GradScaler(enabled=(self.precision == "float16"))

        # Resume from latest checkpoint if available
        latest_ckpt = get_latest_checkpoint(str(self.checkpoint_dir), prefix=checkpoint_prefix)
        if latest_ckpt:
            state = load_checkpoint(latest_ckpt, model, optimizer, map_location=self.device)
            self.global_step = state.get("step", 0)
            self.global_epoch = state.get("epoch", 0)
            self.best_val_loss = state.get("best_val_loss", float("inf"))
            print(f"Resumed from {latest_ckpt} (step={self.global_step}, epoch={self.global_epoch})")

        # Compile model after checkpoint load; keep raw_model for checkpointing
        raw_model = model
        if torch_compile:
            print("Compiling model with torch.compile(mode='max-autotune')...")
            compute_loss = torch.compile(compute_loss, mode="max-autotune")

        epoch = self.global_epoch
        done = False

        while not done:
            if max_epochs is not None and epoch >= max_epochs:
                break

            model.train()
            epoch_loss = 0.0
            num_batches = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)

            for batch in pbar:
                if max_steps is not None and self.global_step >= max_steps:
                    done = True
                    break

                if self.limit_train_batches is not None and num_batches >= self.limit_train_batches:
                    break

                # Move batch to device
                batch = self._to_device(batch)

                optimizer.zero_grad()
                with torch.amp.autocast("cuda", dtype=self.autocast_dtype):
                    loss, metrics = compute_loss(model, batch)

                scaler.scale(loss).backward()

                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                scaler.step(optimizer)
                scaler.update()

                if scheduler is not None:
                    scheduler.step()

                self.global_step += 1
                epoch_loss += loss.item()
                num_batches += 1

                # Progress bar
                pbar.set_postfix(loss=f"{loss.item():.4f}")

                # Logging
                if self.wandb_enabled and self.global_step % log_frequency == 0:
                    import wandb
                    log_dict = {"train/loss": loss.item(), "train/step": self.global_step}
                    log_dict.update({f"train/{k}": v for k, v in metrics.items()})
                    if scheduler is not None:
                        log_dict["train/lr"] = scheduler.get_last_lr()[0]
                    wandb.log(log_dict, step=self.global_step)

                # Step-based validation
                if (val_loader is not None and val_unit == "step"
                        and self.global_step % val_frequency == 0):
                    val_loss, val_metrics = self.validate(model, val_loader, compute_loss)
                    if self.wandb_enabled:
                        import wandb
                        log_dict = {"val/loss": val_loss, "val/step": self.global_step}
                        log_dict.update({f"val/{k}": v for k, v in val_metrics.items()})
                        wandb.log(log_dict, step=self.global_step)
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        save_checkpoint(
                            raw_model, optimizer, self.global_step, epoch,
                            str(self.checkpoint_dir / f"{checkpoint_prefix}_best.pt"),
                            extra={"best_val_loss": self.best_val_loss},
                        )
                    if extra_val_fn is not None:
                        extra_val_fn(model, val_loss, self.global_step, epoch)
                    model.train()

                # Extra logging (e.g., images)
                if extra_log_fn is not None and self.global_step % (log_frequency * 1000) == 0:
                    extra_log_fn(model, batch, self.global_step)

            epoch += 1
            self.global_epoch = epoch

            # Epoch-based validation
            if val_loader is not None and val_unit == "epoch" and epoch % val_frequency == 0:
                val_loss, val_metrics = self.validate(model, val_loader, compute_loss)
                if self.wandb_enabled:
                    import wandb
                    log_dict = {"val/loss": val_loss, "val/epoch": epoch}
                    log_dict.update({f"val/{k}": v for k, v in val_metrics.items()})
                    wandb.log(log_dict, step=self.global_step)

                # Checkpoint on best validation
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    save_checkpoint(
                        raw_model, optimizer, self.global_step, epoch,
                        str(self.checkpoint_dir / f"{checkpoint_prefix}_best.pt"),
                        extra={"best_val_loss": self.best_val_loss},
                    )
                if extra_val_fn is not None:
                    extra_val_fn(model, val_loss, self.global_step, epoch)

            # Periodic checkpoint
            save_checkpoint(
                raw_model, optimizer, self.global_step, epoch,
                str(self.checkpoint_dir / f"{checkpoint_prefix}_latest.pt"),
                extra={"best_val_loss": self.best_val_loss},
            )

        return {"final_step": self.global_step, "final_epoch": self.global_epoch}

    @torch.no_grad()
    def validate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        compute_loss: Callable,
    ) -> tuple[float, dict]:
        """Run validation and return (mean loss, mean metrics dict)."""
        model.eval()
        total_loss = 0.0
        total_metrics: Dict[str, float] = {}
        num_batches = 0

        for batch in tqdm(val_loader, desc="Validation", leave=False):
            if self.limit_val_batches is not None and num_batches >= self.limit_val_batches:
                break
            batch = self._to_device(batch)
            with torch.amp.autocast("cuda", dtype=self.autocast_dtype):
                loss, metrics = compute_loss(model, batch)
            total_loss += loss.item()
            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0.0) + v
            num_batches += 1

        n = max(num_batches, 1)
        model.train()
        return total_loss / n, {k: v / n for k, v in total_metrics.items()}

    def _to_device(self, batch):
        """Move batch to device, handling TensorDict and dicts."""
        if hasattr(batch, "to"):
            return batch.to(self.device)
        elif isinstance(batch, dict):
            return {k: v.to(self.device) if hasattr(v, "to") else v for k, v in batch.items()}
        return batch
