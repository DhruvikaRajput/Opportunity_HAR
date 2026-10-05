"""Command-line experiment runner for Opportunity_HAR research workflows.

Usage:
    python -m src.experiments.run_experiment --config configs/synthetic_cnn.yaml
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_config, save_config
from src.utils.seed import seed_everything
from src.utils.device import get_device, print_device_info
from src.utils.logging import get_logger
from src.data.synthetic import generate_synthetic_imu_data
from src.data.preprocessing import SensorStandardScaler, SensorMinMaxScaler
from src.data.windowing import window_dataframe_by_group
from src.data.splitting import subject_split
from src.data.loader import create_dataloaders
from src.models.baseline import Baseline1DCNN
from src.models.baseline_cnn import BaselineCNN
from src.models.simple_baseline import SimpleLinearBaseline, SimpleMLPBaseline
from src.training.trainer import Trainer
from src.training.evaluate import evaluate_model
from src.visualization.plots import save_all_experiment_plots


def run_experiment(config_path: str, override_device: str = None) -> dict:
    """Execute an experiment run driven entirely by configuration settings.

    Args:
        config_path (str): Path to experiment YAML config.
        override_device (str): Optional compute device override ('cpu', 'cuda').

    Returns:
        dict: Execution summary with test metrics.
    """
    config = load_config(config_path)

    exp_name = config.get("experiment_name", "experiment")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Destination directories
    out_dir_setting = config.get("output_directory") or config.get("paths", {}).get("results_dir", f"results/{exp_name}")
    exp_results_dir = Path(out_dir_setting)
    exp_results_dir.mkdir(parents=True, exist_ok=True)

    checkpoints_dir = Path(config.get("checkpoints_directory") or config.get("paths", {}).get("checkpoints_dir", "checkpoints"))
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Logger
    logger = get_logger(
        name=exp_name,
        log_dir=exp_results_dir,
        log_filename="experiment.log",
    )
    logger.info("Starting Experiment Run: %s (%s)", exp_name, timestamp)
    logger.info("Loaded configuration from: %s", config_path)

    # 1. Reproducibility
    seed = config.get("seed", config.get("reproducibility", {}).get("seed", 42))
    deterministic = config.get("deterministic", config.get("reproducibility", {}).get("deterministic", True))
    seed_everything(seed=seed, deterministic=deterministic)
    logger.info("Enforced random seed: %d (deterministic=%s)", seed, deterministic)

    # 2. Hardware Compute Device & Diagnostic Startup Print
    pref_device = override_device or config.get("device") or config.get("compute", {}).get("device", "auto")
    device = get_device(pref_device)
    dev_info = print_device_info(device)
    print("=" * 60)
    print(f"Device: {device}")
    print(f"GPU name if available: {dev_info.get('device_name', 'N/A')}")
    print(f"CUDA available: {dev_info.get('cuda_available', False)}")
    print("=" * 60)

    # 3. Data Pipeline
    data_cfg = config.get("dataset") or config.get("data", {})
    if data_cfg.get("is_synthetic", True) or data_cfg.get("use_synthetic", True):
        logger.info("DISCLAIMER: Generating SYNTHETIC TEST DATA — NOT OPPORTUNITY")
        num_subjects = data_cfg.get("number_of_subjects", data_cfg.get("num_subjects", 3))
        num_channels = data_cfg.get("number_of_channels", data_cfg.get("num_channels", 18))
        samples_per_subject = data_cfg.get("samples_per_subject", 3000)
        num_classes = data_cfg.get("number_of_classes", data_cfg.get("num_classes", 5))
        sampling_rate = data_cfg.get("sampling_rate_hz", 30.0)
        noise_std = data_cfg.get("noise_std", 0.08)

        raw_df, meta = generate_synthetic_imu_data(
            num_subjects=num_subjects,
            num_channels=num_channels,
            samples_per_subject=samples_per_subject,
            num_classes=num_classes,
            sampling_rate_hz=sampling_rate,
            noise_std=noise_std,
            random_seed=seed,
        )
        channel_cols = meta["channel_names"]
    else:
        raise NotImplementedError(
            "OPPORTUNITY raw dataset loading will be integrated once the download is verified."
        )

    # Subject Split (Strictly prevent data leakage)
    train_subs = data_cfg.get("train_subjects", ["Subject_01"])
    val_subs = data_cfg.get("val_subjects", ["Subject_02"])
    test_subs = data_cfg.get("test_subjects", ["Subject_03"])

    logger.info("Subject partition - Train: %s | Val: %s | Test: %s", train_subs, val_subs, test_subs)
    partitions = subject_split(
        raw_df,
        subject_column="subject_id",
        train_subjects=train_subs,
        val_subjects=val_subs,
        test_subjects=test_subs,
    )

    train_df = partitions["train"]
    val_df = partitions.get("val")
    test_df = partitions.get("test")

    # Preprocessing & Normalization: Fit ONLY on training partition!
    prep_cfg = config.get("preprocessing", {})
    scaler_type = prep_cfg.get("scaler", "standard")
    if scaler_type == "standard":
        scaler = SensorStandardScaler()
    elif scaler_type == "minmax":
        scaler = SensorMinMaxScaler()
    else:
        scaler = None

    if scaler is not None:
        logger.info("Fitting scaler [%s] strictly on training observations...", scaler_type)
        train_df[channel_cols] = scaler.fit_transform(train_df[channel_cols])
        if val_df is not None:
            val_df[channel_cols] = scaler.transform(val_df[channel_cols])
        if test_df is not None:
            test_df[channel_cols] = scaler.transform(test_df[channel_cols])

    # Sliding-Window Segmentation
    win_size = prep_cfg.get("window_length", prep_cfg.get("window_size_samples", 30))
    stride = prep_cfg.get("stride", prep_cfg.get("stride_samples", 15))
    lbl_strat = prep_cfg.get("label_strategy", "mode")

    logger.info("Extracting sliding windows: window_length=%d, stride=%d, strategy='%s'", win_size, stride, lbl_strat)
    X_train, y_train = window_dataframe_by_group(
        train_df,
        sensor_columns=channel_cols,
        label_column="activity_label",
        group_column="subject_id",
        window_size=win_size,
        stride=stride,
        label_strategy=lbl_strat,
        channel_first=True,
    )

    X_val, y_val = None, None
    if val_df is not None:
        X_val, y_val = window_dataframe_by_group(
            val_df,
            sensor_columns=channel_cols,
            label_column="activity_label",
            group_column="subject_id",
            window_size=win_size,
            stride=stride,
            label_strategy=lbl_strat,
            channel_first=True,
        )

    X_test, y_test = None, None
    if test_df is not None:
        X_test, y_test = window_dataframe_by_group(
            test_df,
            sensor_columns=channel_cols,
            label_column="activity_label",
            group_column="subject_id",
            window_size=win_size,
            stride=stride,
            label_strategy=lbl_strat,
            channel_first=True,
        )

    logger.info(
        "Windowed shapes: Train=%s, Val=%s, Test=%s",
        X_train.shape,
        getattr(X_val, "shape", None),
        getattr(X_test, "shape", None),
    )

    # DataLoaders
    train_cfg = config.get("training", {})
    batch_size = train_cfg.get("batch_size", 32)
    num_workers = config.get("num_workers", config.get("compute", {}).get("num_workers", 0))
    pin_memory = config.get("pin_memory", config.get("compute", {}).get("pin_memory", False))

    train_loader, val_loader, test_loader = create_dataloaders(
        train_windows=X_train,
        train_labels=y_train,
        val_windows=X_val,
        val_labels=y_val,
        test_windows=X_test,
        test_labels=y_test,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # 4. Model Construction
    model_cfg = config.get("model", {})
    model_type = model_cfg.get("type", "Baseline1DCNN")

    if model_type == "Baseline1DCNN":
        conv_channels = model_cfg.get("conv_channels", [32, 64, 128])
        kernel_size = model_cfg.get("kernel_size", 5)
        dropout = model_cfg.get("dropout", 0.2)
        model = Baseline1DCNN(
            in_channels=len(channel_cols),
            num_classes=num_classes,
            sequence_length=win_size,
            conv_channels=conv_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )
    elif model_type == "BaselineCNN":
        conv_filters = model_cfg.get("conv_filters", [32, 64, 64])
        kernel_size = model_cfg.get("kernel_size", 5)
        dropout = model_cfg.get("dropout", 0.2)
        model = BaselineCNN(
            in_channels=len(channel_cols),
            num_classes=num_classes,
            conv_filters=conv_filters,
            kernel_size=kernel_size,
            dropout=dropout,
        )
    elif model_type == "SimpleLinearBaseline":
        model = SimpleLinearBaseline(
            in_channels=len(channel_cols),
            sequence_length=win_size,
            num_classes=num_classes,
        )
    elif model_type == "SimpleMLPBaseline":
        hidden_dim = model_cfg.get("hidden_dim", 64)
        dropout = model_cfg.get("dropout", 0.2)
        model = SimpleMLPBaseline(
            in_channels=len(channel_cols),
            sequence_length=win_size,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown model type: '{model_type}'")

    logger.info("Constructed model [%s]: in_channels=%d, num_classes=%d", model_type, len(channel_cols), num_classes)

    # 5. Optimization & Criterion
    lr = train_cfg.get("learning_rate", 0.003)
    wd = train_cfg.get("weight_decay", 0.0001)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()

    # 6. Training with Reusable Trainer
    epochs = train_cfg.get("epochs", 12)
    patience = train_cfg.get("patience", train_cfg.get("early_stopping_patience", 6))
    clip_grad = train_cfg.get("clip_grad_norm", 1.0)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        early_stopping_patience=patience,
        early_stopping_metric="macro_f1",
        checkpoint_dir=checkpoints_dir,
        experiment_name=exp_name,
        clip_grad_norm=clip_grad,
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
    )

    # 7. Model Evaluation Pipeline on Held-out Test Set
    best_checkpoint_path = checkpoints_dir / f"{exp_name}_best.pt"
    test_metrics = evaluate_model(
        model=trainer.model,
        test_loader=test_loader,
        checkpoint_path=best_checkpoint_path if best_checkpoint_path.exists() else None,
        device=device,
        output_dir=exp_results_dir,
        experiment_name=exp_name,
        target_names=[f"Activity_{i}" for i in range(num_classes)],
    )

    logger.info(
        "Final Evaluation: Test Accuracy: %.2f%% | Test Macro F1: %.2f%% | Test Weighted F1: %.2f%%",
        test_metrics["accuracy"] * 100,
        test_metrics["macro_f1"] * 100,
        test_metrics["weighted_f1"] * 100,
    )

    # 8. Save All Visualization Artifacts
    saved_plots = save_all_experiment_plots(
        history=history,
        cm=test_metrics["confusion_matrix"],
        output_dir=exp_results_dir,
        class_names=[f"Act_{i}" for i in range(num_classes)],
    )

    # Save copy of configuration for reproducibility audit
    save_config(config, exp_results_dir / "config.yaml")

    logger.info("Saved all experiment artifacts and plots to: %s", exp_results_dir.resolve())
    for plot_name, plot_file in saved_plots.items():
        logger.info("  Plot [%s]: %s", plot_name, plot_file.name)

    return {
        "experiment_name": exp_name,
        "results_dir": str(exp_results_dir.resolve()),
        "checkpoint_path": str(best_checkpoint_path.resolve()) if best_checkpoint_path.exists() else None,
        "test_metrics": test_metrics,
        "history": history,
        "saved_plots": {k: str(v) for k, v in saved_plots.items()},
    }


def main():
    parser = argparse.ArgumentParser(description="Opportunity_HAR Experiment Runner")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML configuration")
    parser.add_argument("--device", type=str, default=None, help="Device override ('cpu', 'cuda')")
    args = parser.parse_args()

    run_experiment(config_path=args.config, override_device=args.device)


if __name__ == "__main__":
    main()
