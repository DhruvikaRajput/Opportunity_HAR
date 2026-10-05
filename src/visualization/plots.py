"""Research-quality visualization functions for HAR sensor signals, training dynamics, and evaluations.
"""

from typing import List, Optional, Union, Dict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_sensor_signals(
    signals: np.ndarray,
    channel_names: Optional[List[str]] = None,
    sampling_rate_hz: Optional[float] = None,
    title: str = "Multi-Channel Sensor Signals",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot continuous or windowed sensor time-series channels."""
    if signals.shape[0] < signals.shape[1] and channel_names and len(channel_names) == signals.shape[0]:
        sig_data = signals
        num_channels, time_steps = signals.shape
    else:
        sig_data = signals.T
        num_channels, time_steps = sig_data.shape

    time_axis = np.arange(time_steps) / sampling_rate_hz if sampling_rate_hz else np.arange(time_steps)
    x_label = "Time (seconds)" if sampling_rate_hz else "Time step"

    fig, axes = plt.subplots(num_channels, 1, figsize=(10, max(4, 1.5 * num_channels)), sharex=True)
    if num_channels == 1:
        axes = [axes]

    for i in range(num_channels):
        ax = axes[i]
        lbl = channel_names[i] if channel_names and i < len(channel_names) else f"Ch {i}"
        ax.plot(time_axis, sig_data[i], label=lbl, color="tab:blue", linewidth=1.2)
        ax.set_ylabel(lbl, fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)

    axes[-1].set_xlabel(x_label)
    fig.suptitle(title, fontsize=12, y=0.99)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_confusion_matrix(
    cm: Union[np.ndarray, List[List[int]]],
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    title: str = "Confusion Matrix",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot a publication-grade confusion matrix heatmap."""
    cm_arr = np.asarray(cm, dtype=np.float32)
    if normalize:
        row_sums = cm_arr.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        cm_display = cm_arr / row_sums
        fmt = ".2f"
    else:
        cm_display = cm_arr
        fmt = "d"

    fig, ax = plt.subplots(figsize=(6, 5))
    labels = class_names if class_names else [f"C{i}" for i in range(len(cm_arr))]

    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        ax=ax,
    )
    ax.set_ylabel("True Activity", fontsize=10)
    ax.set_xlabel("Predicted Activity", fontsize=10)
    ax.set_title(title, fontsize=12)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_training_curves(
    history: dict,
    title: str = "Training and Validation Dynamics",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot dual loss and validation metric progression curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    epochs = history.get("epoch", range(1, len(history.get("train_loss", [])) + 1))

    if "train_loss" in history and history["train_loss"]:
        ax1.plot(epochs, history["train_loss"], label="Train Loss", color="tab:blue", marker="o", markersize=3)
    if "val_loss" in history and any(v is not None for v in history.get("val_loss", [])):
        valid_val_loss = [v for v in history["val_loss"] if v is not None]
        ax1.plot(epochs[:len(valid_val_loss)], valid_val_loss, label="Val Loss", color="tab:orange", marker="s", markersize=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Cross-Entropy Loss")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    if "val_macro_f1" in history and any(v is not None for v in history.get("val_macro_f1", [])):
        valid_f1 = [v for v in history["val_macro_f1"] if v is not None]
        ax2.plot(epochs[:len(valid_f1)], valid_f1, label="Val Macro F1", color="tab:green", marker="^", markersize=3)
    if "val_accuracy" in history and any(v is not None for v in history.get("val_accuracy", [])):
        valid_acc = [v for v in history["val_accuracy"] if v is not None]
        ax2.plot(epochs[:len(valid_acc)], valid_acc, label="Val Accuracy", color="tab:purple", marker="d", markersize=3)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_title("Validation Metrics")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_single_metric(
    epochs: List[int],
    values: List[float],
    title: str,
    ylabel: str,
    color: str = "tab:blue",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot a single metric curve across epochs and save to disk."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, values, color=color, marker="o", markersize=3, linewidth=1.5)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig


def save_all_experiment_plots(
    history: Dict[str, list],
    cm: Union[np.ndarray, List[List[int]]],
    output_dir: Union[str, Path],
    class_names: Optional[List[str]] = None,
) -> Dict[str, Path]:
    """Generate and save the required individual research plots:
        - training_loss.png
        - validation_loss.png
        - training_accuracy.png
        - validation_accuracy.png
        - confusion_matrix.png

    Args:
        history (Dict[str, list]): History from Trainer containing loss and accuracy lists.
        cm (Union[np.ndarray, List[List[int]]]): Confusion matrix array.
        output_dir (Union[str, Path]): Target output directory.
        class_names (Optional[List[str]]): Target activity class names.

    Returns:
        Dict[str, Path]: Paths to saved image artifacts.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    epochs = history.get("epoch", list(range(1, len(history.get("train_loss", [])) + 1)))

    saved_paths = {}

    # 1. training_loss.png
    if "train_loss" in history and history["train_loss"]:
        p = out_path / "training_loss.png"
        plot_single_metric(epochs, history["train_loss"], "Training Loss", "Loss", color="tab:blue", save_path=p)
        saved_paths["training_loss"] = p

    # 2. validation_loss.png
    if "val_loss" in history and any(v is not None for v in history["val_loss"]):
        val_losses = [v for v in history["val_loss"] if v is not None]
        p = out_path / "validation_loss.png"
        plot_single_metric(epochs[:len(val_losses)], val_losses, "Validation Loss", "Loss", color="tab:orange", save_path=p)
        saved_paths["validation_loss"] = p

    # 3. training_accuracy.png
    if "train_accuracy" in history and history["train_accuracy"]:
        p = out_path / "training_accuracy.png"
        plot_single_metric(epochs, history["train_accuracy"], "Training Accuracy", "Accuracy", color="tab:green", save_path=p)
        saved_paths["training_accuracy"] = p

    # 4. validation_accuracy.png
    if "val_accuracy" in history and any(v is not None for v in history["val_accuracy"]):
        val_accs = [v for v in history["val_accuracy"] if v is not None]
        p = out_path / "validation_accuracy.png"
        plot_single_metric(epochs[:len(val_accs)], val_accs, "Validation Accuracy", "Accuracy", color="tab:purple", save_path=p)
        saved_paths["validation_accuracy"] = p

    # 5. confusion_matrix.png
    p = out_path / "confusion_matrix.png"
    plot_confusion_matrix(cm, class_names=class_names, title="Test Set Confusion Matrix", save_path=p)
    saved_paths["confusion_matrix"] = p

    return saved_paths
