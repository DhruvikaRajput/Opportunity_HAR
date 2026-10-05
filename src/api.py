"""High-level Notebook API for the OPPORTUNITY Human Activity Recognition Project.

Provides a unified, user-friendly interface designed for Google Colab and interactive notebooks.
Notebooks should call this API rather than importing deep internal modules directly.

Supports three experiment modes:
    - 'DEBUG': Fast 1-batch/1-epoch smoke test on CPU or GPU for instant pipeline verification.
    - 'FAST' : Moderate training (e.g., 5 epochs) requiring CUDA GPU.
    - 'FULL' : Full research training (50+ epochs, early stopping) requiring CUDA GPU.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.seed import seed_everything
from src.utils.device import get_device, print_device_info as get_device_info
from src.utils.logging import get_logger
from src.data.opportunity_loader import OpportunityLoader
from src.data.preprocessing import SensorStandardScaler
from src.data.windowing import create_sliding_windows
from src.data.loader import HARDataset, create_dataloaders as _create_dataloaders
from src.data.preprocess_opportunity import (
    preprocess_opportunity_dataset,
    remap_labels_to_continuous_indices,
)
from src.features.body_regions import (
    BODY_REGIONS,
    ANATOMICAL_SENSOR_COLUMNS,
    get_fixed_adjacency_matrix,
    plot_body_region_graph,
)
from src.models.classical import (
    extract_statistical_features,
    train_random_forest,
    train_logistic_regression,
    train_linear_svm,
)
from src.models.cnn import OpportunityCNN
from src.models.fixed_gnn import FixedAnatomicalGNN
from src.models.adaptive_gnn import ActivityAdaptiveAnatomicalGNN
from src.training.engine import train_model as _engine_train_model, evaluate_epoch as _engine_evaluate_epoch
from src.evaluation.metrics import compute_har_metrics, metrics_summary_table
from src.visualization.plots import (
    plot_confusion_matrix,
    plot_training_curves,
)

logger = get_logger("api")


def setup_project(seed: int = 42) -> Dict[str, Any]:
    """Initialize environment, set random seed, verify directories, and report compute device.

    Args:
        seed (int): Global random seed.

    Returns:
        Dict[str, Any]: Environment status report.
    """
    seed_everything(seed)
    dev_info = get_device_info()

    # Ensure required runtime directories exist
    for folder in ["checkpoints", "results", "data/processed"]:
        (REPO_ROOT / folder).mkdir(parents=True, exist_ok=True)

    is_colab = "google.colab" in sys.modules
    status = {
        "repo_root": str(REPO_ROOT),
        "is_colab": is_colab,
        "device": dev_info["device_type"],
        "cuda_available": dev_info["cuda_available"],
        "device_name": dev_info["device_name"],
        "seed": seed,
    }

    logger.info("Project initialized. Device: %s (%s). Colab: %s",
                status["device"], status["device_name"], status["is_colab"])
    return status


def load_opportunity_data(
    subject: str = "S1",
    run: str = "ADL1",
    target_track: str = "Locomotion",
    sensor_selection: str = "on_body",
    interpolate_nans: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load continuous raw recording for a specific subject and run.

    Args:
        subject (str): Subject ID ('S1', 'S2', 'S3', 'S4').
        run (str): Run ID ('ADL1'-'ADL5', 'Drill').
        target_track (str): Label track ('Locomotion' or 'Gestures').
        sensor_selection (str): 'on_body' (133 channels) or 'all' (242 channels).
        interpolate_nans (bool): Whether to perform linear missing value interpolation.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (sensor_data, labels, timestamps).
    """
    loader = OpportunityLoader(target_track=target_track, sensor_selection=sensor_selection)
    return loader.load_recording(subject, run, interpolate_nans=interpolate_nans)


def preprocess_data(
    track: str = "Locomotion",
    output_filename: str = "opportunity_locomotion_subset.npz",
    window_size: int = 30,
    stride: int = 15,
) -> Dict[str, Any]:
    """Execute leak-free end-to-end preprocessing on OPPORTUNITY recordings.

    Args:
        track (str): 'Locomotion' or 'ML_Both_Arms'.
        output_filename (str): Name of output .npz file under data/processed/.
        window_size (int): Window duration in timesteps (30 steps = 1.0 s at 30 Hz).
        stride (int): Window stride in timesteps (15 steps = 50% overlap).

    Returns:
        Dict[str, Any]: Summary dictionary with window and class counts.
    """
    saved_path = preprocess_opportunity_dataset(
        target_track=track,
        window_length=window_size,
        stride=stride,
        output_filename=output_filename,
    )
    import json
    data = np.load(saved_path, allow_pickle=True)
    name_map = json.loads(str(data["class_names"])) if "class_names" in data else {}
    total_w = len(data["X_train"]) + len(data["X_val"]) + len(data["X_test"])
    class_dist = {
        name_map.get(str(i), f"Class_{i}"): int((data["y_train"] == i).sum())
        for i in np.unique(data["y_train"])
    }
    return {
        "saved_path": str(saved_path),
        "total_windows": total_w,
        "train_windows": len(data["X_train"]),
        "val_windows": len(data["X_val"]),
        "test_windows": len(data["X_test"]),
        "class_distribution": class_dist,
    }


def load_preprocessed_subset(filename: str = "opportunity_locomotion_subset.npz") -> Dict[str, Any]:
    """Load preprocessed windowed arrays from data/processed/.

    Returns:
        Dict[str, Any]: Contains X_train, y_train, X_val, y_val, X_test, y_test, class_names.
    """
    path = REPO_ROOT / "data" / "processed" / filename
    if not path.exists():
        raise FileNotFoundError(f"Processed file not found at {path}. Run preprocess_data() first.")
    data = np.load(path, allow_pickle=True)
    out = {k: data[k] for k in data.files}
    import json
    if "class_names" in out:
        try:
            name_dict = json.loads(str(out["class_names"]))
            max_idx = max(int(k) for k in name_dict.keys())
            name_list = [name_dict.get(str(i), name_dict.get(i, f"Class_{i}")) for i in range(max_idx + 1)]
            out["class_names"] = name_list
        except Exception:
            pass
    return out


def create_windows(
    data: np.ndarray,
    labels: np.ndarray,
    window_size: int = 30,
    stride: int = 15,
    channel_first: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Segment continuous time-series into sliding windows."""
    return create_sliding_windows(data, labels, window_size=window_size, stride=stride, channel_first=channel_first)


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    batch_size: int = 64,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """Construct PyTorch DataLoaders for train, validation, and test splits."""
    return _create_dataloaders(
        train_windows=X_train,
        train_labels=y_train,
        val_windows=X_val,
        val_labels=y_val,
        test_windows=X_test,
        test_labels=y_test,
        batch_size=batch_size,
    )


def build_cnn(
    in_channels: int = 133,
    num_classes: int = 5,
    sequence_length: int = 30,
    conv_channels: Optional[List[int]] = None,
    kernel_size: int = 5,
    dropout: float = 0.3,
) -> nn.Module:
    """Build 1D CNN baseline model."""
    return OpportunityCNN(
        in_channels=in_channels,
        num_classes=num_classes,
        sequence_length=sequence_length,
        conv_channels=conv_channels,
        kernel_size=kernel_size,
        dropout=dropout,
    )


def build_fixed_gnn(
    num_classes: int = 5,
    sequence_length: int = 30,
    node_dim: int = 64,
    gcn_hidden_dim: int = 64,
    dropout: float = 0.2,
) -> nn.Module:
    """Build Fixed Anatomical GNN model."""
    return FixedAnatomicalGNN(
        num_classes=num_classes,
        sequence_length=sequence_length,
        node_dim=node_dim,
        gcn_hidden_dim=gcn_hidden_dim,
        dropout=dropout,
    )


def build_adaptive_gnn(
    num_classes: int = 5,
    sequence_length: int = 30,
    node_dim: int = 64,
    gcn_hidden_dim: int = 64,
    dropout: float = 0.2,
) -> nn.Module:
    """Build Activity-Adaptive Anatomical GNN model."""
    return ActivityAdaptiveAnatomicalGNN(
        num_classes=num_classes,
        sequence_length=sequence_length,
        node_dim=node_dim,
        gcn_hidden_dim=gcn_hidden_dim,
        dropout=dropout,
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    mode: str = "DEBUG",
    epochs: Optional[int] = None,
    lr: float = 1e-3,
    experiment_name: str = "experiment",
    checkpoint_dir: Optional[Union[str, Path]] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Train a neural network model adhering to experiment mode and compute guards.

    Modes:
        - 'DEBUG': 1-epoch / 1-batch smoke test. Executes on CPU or GPU for instant verification.
        - 'FAST' : 5 epochs on CUDA GPU. (Stops if CUDA is unavailable).
        - 'FULL' : 50 epochs with early stopping on CUDA GPU. (Stops if CUDA is unavailable).

    Args:
        model (nn.Module): Neural network.
        train_loader (DataLoader): Training loader.
        val_loader (Optional[DataLoader]): Validation loader.
        mode (str): 'DEBUG', 'FAST', or 'FULL'.
        epochs (Optional[int]): Override epoch count if specified.
        lr (float): Learning rate.
        experiment_name (str): Name for saved checkpoint weights.
        checkpoint_dir (Optional[Union[str, Path]]): Destination for checkpoints.

    Returns:
        Tuple[nn.Module, Dict[str, Any]]: (best_model, training_history).
    """
    mode = mode.upper()
    cuda_avail = torch.cuda.is_available()

    if mode in ["FAST", "FULL"] and not cuda_avail:
        raise RuntimeError(
            f"\n[CUDA REQUIRED - COMPUTE RULE ENFORCED]\n"
            f"Training in '{mode}' mode requires a CUDA GPU to prevent freezing your laptop CPU.\n"
            f"Please switch your Google Colab runtime to GPU (Runtime > Change runtime type > T4 GPU),\n"
            f"or set EXPERIMENT_MODE = 'DEBUG' to run a fast 1-batch CPU verification test."
        )

    # Determine device
    device = torch.device("cuda" if cuda_avail else "cpu")

    # Set epochs based on mode
    if epochs is None:
        if mode == "DEBUG":
            epochs = 1
        elif mode == "FAST":
            epochs = 5
        else:  # FULL
            epochs = 50

    patience = 2 if mode == "FAST" else 10
    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else (REPO_ROOT / "checkpoints")

    logger.info("Starting training: Model=%s, Mode=%s, Epochs=%d, Device=%s",
                model.__class__.__name__, mode, epochs, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_model, history = _engine_train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        scheduler=scheduler,
        early_stopping_patience=patience,
        checkpoint_dir=ckpt_dir,
        experiment_name=experiment_name,
    )

    return best_model, history


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Evaluate trained model on test data and compute full HAR metrics."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.CrossEntropyLoss()
    loss, metrics = _engine_evaluate_epoch(model, test_loader, criterion, device)
    metrics["test_loss"] = loss
    return metrics


def run_classical_baselines() -> Dict[str, Any]:
    """Run Classical ML Baselines (Random Forest, Logistic Regression, Linear SVM) on CPU."""
    from src.experiments.run_baseline import main as _run_base
    return _run_base()


def run_loso(model: str = "random_forest", max_runs_per_sub: int = 2) -> Any:
    """Run Leave-One-Subject-Out (LOSO) cross-validation."""
    from src.experiments.run_loso import run_classical_loso
    out_dir = REPO_ROOT / "results" / "loso"
    return run_classical_loso(out_dir, max_runs_per_sub=max_runs_per_sub)


def run_ablation(model_type: str = "random_forest") -> Any:
    """Run anatomical body-region ablation experiments."""
    from src.experiments.run_ablation import run_ablation_experiments
    out_dir = REPO_ROOT / "results" / "body_ablation"
    return run_ablation_experiments(out_dir)


def run_robustness(model_type: str = "random_forest") -> Any:
    """Run sensor failure and missing-modality robustness study."""
    from src.experiments.run_robustness import run_robustness_study
    out_dir = REPO_ROOT / "results" / "robustness"
    return run_robustness_study(out_dir)
