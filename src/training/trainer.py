"""Reusable PyTorch training engine for multivariate time-series HAR models.

Supports automatic CPU/CUDA selection, early stopping, checkpointing, and metric logging.
Works with any compatible PyTorch model architecture.
"""

from typing import Dict, Any, Optional, Tuple, Union
from pathlib import Path
import copy
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from src.utils.device import get_device, print_device_info
from src.evaluation.metrics import compute_har_metrics

logger = logging.getLogger(__name__)


class Trainer:
    """Generic, reusable PyTorch model trainer for HAR research workflows.

    Features:
        - Automatic CPU and CUDA GPU detection and device dispatch.
        - Early stopping based on validation metrics (Macro F1 or Loss).
        - State-dict checkpointing for best model persistence.
        - Detailed epoch history tracking for loss and accuracy curves.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: Optional[nn.Module] = None,
        device: Optional[Union[str, torch.device]] = None,
        scheduler: Optional[_LRScheduler] = None,
        early_stopping_patience: int = 10,
        early_stopping_metric: str = "macro_f1",  # 'macro_f1' or 'loss'
        checkpoint_dir: Optional[Union[str, Path]] = None,
        experiment_name: str = "model",
        clip_grad_norm: Optional[float] = 1.0,
    ):
        """Initialize the Trainer instance.

        Args:
            model (nn.Module): Any PyTorch neural network model.
            optimizer (Optimizer): PyTorch optimizer.
            criterion (Optional[nn.Module]): Loss criterion. Defaults to CrossEntropyLoss.
            device (Optional[Union[str, torch.device]]): Target device or 'auto'.
            scheduler (Optional[_LRScheduler]): Learning rate scheduler.
            early_stopping_patience (int): Epochs to tolerate without metric improvement.
            early_stopping_metric (str): Metric to monitor ('macro_f1' or 'loss').
            checkpoint_dir (Optional[Union[str, Path]]): Path to persist best checkpoint weights.
            experiment_name (str): Identifier for saved checkpoint filenames.
            clip_grad_norm (Optional[float]): Gradient norm clipping threshold.
        """
        # 1. Resolve Device
        if device is None or isinstance(device, str):
            self.device = get_device(device)
        else:
            self.device = device

        # Print runtime device diagnostics as requested
        dev_info = print_device_info(self.device)
        print("=" * 50)
        print(f"Device: {self.device}")
        print(f"GPU name if available: {dev_info.get('device_name', 'N/A')}")
        print(f"CUDA available: {dev_info.get('cuda_available', False)}")
        print("=" * 50)

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.scheduler = scheduler
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_metric = early_stopping_metric
        self.clip_grad_norm = clip_grad_norm
        self.experiment_name = experiment_name

        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.history: Dict[str, list] = {
            "epoch": [],
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_macro_f1": [],
            "lr": [],
        }

        self.best_metric_val: Optional[float] = None
        self.best_model_state: Optional[dict] = None
        self.best_checkpoint_path: Optional[Path] = None

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Execute one complete training epoch over mini-batches.

        Args:
            train_loader (DataLoader): DataLoader for training data.

        Returns:
            Tuple[float, float]: (average_loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct_samples = 0
        total_samples = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(x_batch)
            loss = self.criterion(logits, y_batch)
            loss.backward()

            if self.clip_grad_norm is not None and self.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm)

            self.optimizer.step()

            batch_size = x_batch.size(0)
            running_loss += loss.item() * batch_size
            preds = torch.argmax(logits, dim=1)
            correct_samples += (preds == y_batch).sum().item()
            total_samples += batch_size

        epoch_loss = running_loss / max(1, total_samples)
        epoch_acc = correct_samples / max(1, total_samples)
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, Dict[str, Any]]:
        """Evaluate model across validation mini-batches.

        Args:
            val_loader (DataLoader): DataLoader for validation data.

        Returns:
            Tuple[float, Dict[str, Any]]: (average_loss, metrics_dict)
        """
        self.model.eval()
        running_loss = 0.0
        total_samples = 0
        all_preds = []
        all_trues = []

        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            logits = self.model(x_batch)
            loss = self.criterion(logits, y_batch)

            batch_size = x_batch.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_trues.extend(y_batch.cpu().numpy().tolist())

        epoch_loss = running_loss / max(1, total_samples)
        metrics = compute_har_metrics(all_trues, all_preds)
        return epoch_loss, metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 20,
    ) -> Dict[str, list]:
        """Run the full training procedure across multiple epochs.

        Args:
            train_loader (DataLoader): Training DataLoader.
            val_loader (Optional[DataLoader]): Validation DataLoader.
            epochs (int): Total training epochs.

        Returns:
            Dict[str, list]: Training history dictionary.
        """
        logger.info("Starting model training for %d epochs...", epochs)
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader)
            current_lr = self.optimizer.param_groups[0]["lr"]

            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["train_accuracy"].append(train_acc)
            self.history["lr"].append(current_lr)

            log_line = f"Epoch [{epoch:03d}/{epochs:03d}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%"

            if val_loader is not None:
                val_loss, val_metrics = self.validate_epoch(val_loader)
                val_acc = val_metrics["accuracy"]
                val_f1 = val_metrics["macro_f1"]

                self.history["val_loss"].append(val_loss)
                self.history["val_accuracy"].append(val_acc)
                self.history["val_macro_f1"].append(val_f1)

                log_line += f" | Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%, Val Macro F1: {val_f1*100:.2f}%"

                # Check improvement
                monitored = val_f1 if self.early_stopping_metric == "macro_f1" else -val_loss
                if self.best_metric_val is None or monitored > self.best_metric_val:
                    self.best_metric_val = monitored
                    self.best_model_state = copy.deepcopy(self.model.state_dict())
                    patience_counter = 0

                    if self.checkpoint_dir:
                        ckpt_file = self.checkpoint_dir / f"{self.experiment_name}_best.pt"
                        self.save_checkpoint(ckpt_file, epoch=epoch, metrics=val_metrics)
                        self.best_checkpoint_path = ckpt_file
                        log_line += " [* Best Checkpoint Saved]"
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        logger.info(log_line)
                        logger.info(
                            "Early stopping triggered at epoch %d (patience = %d)",
                            epoch,
                            self.early_stopping_patience,
                        )
                        break
            else:
                self.history["val_loss"].append(None)
                self.history["val_accuracy"].append(None)
                self.history["val_macro_f1"].append(None)

            logger.info(log_line)

            if self.scheduler is not None:
                self.scheduler.step()

        # Restore best weights if validation was performed
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info("Restored model to best validation checkpoint weights.")

        return self.history

    def save_checkpoint(
        self,
        filepath: Union[str, Path],
        epoch: int,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Persist model and optimizer state dictionaries to a file.

        Args:
            filepath (Union[str, Path]): Destination file path.
            epoch (int): Epoch number.
            metrics (Optional[Dict[str, Any]]): Accompanying metrics dictionary.

        Returns:
            Path: Absolute path to saved checkpoint.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics or {},
        }
        torch.save(payload, path)
        return path

    def load_checkpoint(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """Load state dictionary from a saved checkpoint file.

        Args:
            filepath (Union[str, Path]): Path to checkpoint file.

        Returns:
            Dict[str, Any]: Checkpoint payload dictionary.
        """
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {path.resolve()}")

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info("Successfully loaded checkpoint from: %s", path)
        return checkpoint
