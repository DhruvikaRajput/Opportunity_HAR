"""Model evaluation and inference pipeline.

Provides functions to load trained model checkpoints, execute forward inference,
compute standardized HAR metrics, and persist outputs in JSON/CSV formats.
"""

from typing import Dict, Any, Optional, Tuple, Union, List
from pathlib import Path
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from src.utils.device import get_device
from src.evaluation.metrics import compute_har_metrics, save_metrics_to_json, metrics_summary_table


@torch.no_grad()
def run_inference(
    model: nn.Module,
    data_loader: DataLoader,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference over a DataLoader and collect predictions, probabilities, and targets.

    Args:
        model (nn.Module): PyTorch model.
        data_loader (DataLoader): DataLoader to evaluate.
        device (Optional[Union[str, torch.device]]): Target device.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - y_true: Ground truth labels (N,).
            - y_pred: Predicted class labels (N,).
            - y_prob: Softmax class probabilities (N, num_classes).
    """
    if device is None or isinstance(device, str):
        dev = get_device(device)
    else:
        dev = device

    model = model.to(dev)
    model.eval()

    all_preds = []
    all_trues = []
    all_probs = []

    for x_batch, y_batch in data_loader:
        x_batch = x_batch.to(dev)
        logits = model(x_batch)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_trues.extend(y_batch.numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())

    return np.array(all_trues), np.array(all_preds), np.array(all_probs)


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    experiment_name: str = "evaluation",
    target_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Evaluate a PyTorch model (optionally loading from checkpoint) and persist metrics.

    Args:
        model (nn.Module): Model architecture.
        test_loader (DataLoader): Evaluation DataLoader.
        checkpoint_path (Optional[Union[str, Path]]): Path to saved checkpoint file.
        device (Optional[Union[str, torch.device]]): Computation device.
        output_dir (Optional[Union[str, Path]]): Directory to save metrics.json and metrics.csv.
        experiment_name (str): Label for the evaluation run.
        target_names (Optional[List[str]]): Optional human-readable class names.

    Returns:
        Dict[str, Any]: Evaluated metrics dictionary.
    """
    if device is None or isinstance(device, str):
        dev = get_device(device)
    else:
        dev = device

    if checkpoint_path is not None:
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path.resolve()}")
        checkpoint = torch.load(ckpt_path, map_location=dev)
        model.load_state_dict(checkpoint["model_state_dict"])

    y_true, y_pred, y_prob = run_inference(model, test_loader, device=dev)
    metrics = compute_har_metrics(y_true, y_pred, target_names=target_names)

    # Save artifacts if output_dir specified
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # 1. metrics.json
        save_metrics_to_json(metrics, out_path / "metrics.json")

        # 2. metrics.csv
        summary_df = metrics_summary_table([{
            "experiment_name": experiment_name,
            **metrics,
        }])
        summary_df.to_csv(out_path / "metrics.csv", index=False)

        # 3. Save raw predictions
        preds_df = pd.DataFrame({
            "y_true": y_true,
            "y_pred": y_pred,
        })
        preds_df.to_csv(out_path / "predictions.csv", index=False)

    return metrics
