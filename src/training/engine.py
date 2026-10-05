"""Training loop, validation, early stopping, and checkpointing engine.
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

from src.evaluation.metrics import compute_har_metrics

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to prevent overfitting when validation performance ceases to improve."""

    def __init__(
        self,
        patience: int = 10,
        mode: str = "max",
        min_delta: float = 1e-4,
    ):
        """Initialize EarlyStopping.

        Args:
            patience (int): Number of epochs to wait without improvement before stopping.
            mode (str): 'max' for metrics like macro_f1/accuracy, 'min' for loss.
            min_delta (float): Minimum change to qualify as an improvement.
        """
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop: bool = False
        self.best_state_dict: Optional[dict] = None

    def __call__(self, current_score: float, model: nn.Module) -> bool:
        """Update tracker with the latest validation score.

        Args:
            current_score (float): Current epoch validation metric.
            model (nn.Module): Current model state.

        Returns:
            bool: True if this score is the best observed so far, False otherwise.
        """
        if self.best_score is None:
            self.best_score = current_score
            self.best_state_dict = copy.deepcopy(model.state_dict())
            return True

        if self.mode == "max":
            improved = current_score > (self.best_score + self.min_delta)
        else:
            improved = current_score < (self.best_score - self.min_delta)

        if improved:
            self.best_score = current_score
            self.best_state_dict = copy.deepcopy(model.state_dict())
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    clip_grad_norm: Optional[float] = None,
) -> float:
    """Run one epoch of training over mini-batches.

    Args:
        model (nn.Module): PyTorch neural network.
        loader (DataLoader): Training DataLoader yielding (windows, labels).
        criterion (nn.Module): Loss function (e.g. CrossEntropyLoss).
        optimizer (Optimizer): Optimizer.
        device (torch.device): CPU or CUDA device.
        clip_grad_norm (Optional[float]): Max norm for gradient clipping.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()

        if clip_grad_norm is not None and clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)

        optimizer.step()

        running_loss += loss.item() * len(y_batch)
        total_samples += len(y_batch)

    return running_loss / max(1, total_samples)


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, Any]]:
    """Evaluate model performance across a validation or test DataLoader.

    Args:
        model (nn.Module): PyTorch model.
        loader (DataLoader): Evaluation DataLoader.
        criterion (nn.Module): Loss function.
        device (torch.device): CPU or CUDA device.

    Returns:
        Tuple[float, Dict[str, Any]]: Average evaluation loss and comprehensive HAR metrics dict.
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0
    all_preds = []
    all_trues = []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(x_batch)
        loss = criterion(logits, y_batch)

        running_loss += loss.item() * len(y_batch)
        total_samples += len(y_batch)

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_trues.extend(y_batch.cpu().numpy())

    avg_loss = running_loss / max(1, total_samples)
    metrics = compute_har_metrics(all_trues, all_preds)
    return avg_loss, metrics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    epochs: int = 50,
    scheduler: Optional[_LRScheduler] = None,
    early_stopping_patience: int = 10,
    checkpoint_dir: Optional[Union[str, Path]] = None,
    experiment_name: str = "model",
    clip_grad_norm: Optional[float] = 1.0,
) -> Tuple[nn.Module, Dict[str, list]]:
    """Complete research training loop with early stopping, checkpointing, and metric logging.

    Args:
        model (nn.Module): Neural network model.
        train_loader (DataLoader): DataLoader for training.
        val_loader (Optional[DataLoader]): DataLoader for validation.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer.
        device (torch.device): Target computation device (CPU or CUDA).
        epochs (int): Maximum training epochs.
        scheduler (Optional[_LRScheduler]): Learning rate scheduler.
        early_stopping_patience (int): Epoch patience threshold.
        checkpoint_dir (Optional[Union[str, Path]]): Directory to save best checkpoint weights.
        experiment_name (str): Identifier for saved checkpoint filenames.
        clip_grad_norm (Optional[float]): Gradient norm clipping threshold.

    Returns:
        Tuple[nn.Module, Dict[str, list]]:
            - Best model restored to state with highest validation Macro F1.
            - History dictionary containing epoch-by-epoch tracking data.
    """
    model = model.to(device)
    early_stopper = EarlyStopping(patience=early_stopping_patience, mode="max")

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_macro_f1": [],
        "val_accuracy": [],
    }

    best_checkpoint_path = None
    if checkpoint_dir:
        chk_path = Path(checkpoint_dir)
        chk_path.mkdir(parents=True, exist_ok=True)
        best_checkpoint_path = chk_path / f"{experiment_name}_best.pt"

    logger.info("Initiating training for %d epochs on device: %s", epochs, device)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            clip_grad_norm=clip_grad_norm,
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)

        log_msg = f"Epoch [{epoch:03d}/{epochs:03d}] - Train Loss: {train_loss:.4f}"

        if val_loader is not None:
            val_loss, val_metrics = evaluate_epoch(model, val_loader, criterion, device)
            val_f1 = val_metrics["macro_f1"]
            val_acc = val_metrics["accuracy"]

            history["val_loss"].append(val_loss)
            history["val_macro_f1"].append(val_f1)
            history["val_accuracy"].append(val_acc)

            log_msg += f" | Val Loss: {val_loss:.4f} | Val Macro F1: {val_f1 * 100:.2f}% | Val Acc: {val_acc * 100:.2f}%"

            is_best = early_stopper(val_f1, model)
            if is_best and best_checkpoint_path:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_macro_f1": val_f1,
                        "val_loss": val_loss,
                    },
                    best_checkpoint_path,
                )
                log_msg += " [* Saved Best Checkpoint]"

            if early_stopper.early_stop:
                logger.info(log_msg)
                logger.info("Early stopping triggered at epoch %d (patience = %d)", epoch, early_stopping_patience)
                break
        else:
            history["val_loss"].append(None)
            history["val_macro_f1"].append(None)
            history["val_accuracy"].append(None)

        logger.info(log_msg)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                step_metric = val_metrics.get("macro_f1", 0.0) if val_loader is not None else train_loss
                scheduler.step(step_metric)
            else:
                scheduler.step()

    # Load best checkpoint weights if available
    if early_stopper.best_state_dict is not None:
        model.load_state_dict(early_stopper.best_state_dict)
        logger.info("Restored model weights to optimal validation state (Score: %.4f)", early_stopper.best_score)

    return model, history
