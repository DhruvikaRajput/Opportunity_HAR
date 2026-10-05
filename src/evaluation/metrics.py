"""Evaluation metrics and reporting tools tailored for Human Activity Recognition research.

Provides standardized calculations for Accuracy, Macro F1, Weighted F1,
Macro Precision, Macro Recall, and Confusion Matrices.
"""

from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def compute_har_metrics(
    y_true: Union[np.ndarray, List[int]],
    y_pred: Union[np.ndarray, List[int]],
    labels: Optional[List[int]] = None,
    target_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Calculate standard benchmark metrics for multi-class HAR evaluation.

    Args:
        y_true (Union[np.ndarray, List[int]]): Ground-truth class labels.
        y_pred (Union[np.ndarray, List[int]]): Predicted class labels.
        labels (Optional[List[int]]): Optional subset of class labels to evaluate.
        target_names (Optional[List[str]]): Human-readable names for classes.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'accuracy': float
            - 'macro_f1': float (primary benchmark metric in HAR due to class imbalance)
            - 'weighted_f1': float
            - 'macro_precision': float
            - 'macro_recall': float
            - 'confusion_matrix': List[List[int]]
            - 'classification_report': dict
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, labels=labels, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true_arr, y_pred_arr, labels=labels, average="weighted", zero_division=0))
    macro_prec = float(precision_score(y_true_arr, y_pred_arr, labels=labels, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true_arr, y_pred_arr, labels=labels, average="macro", zero_division=0))

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)

    report = classification_report(
        y_true_arr,
        y_pred_arr,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def save_metrics_to_json(metrics: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """Save metrics dictionary to a JSON file.

    Args:
        metrics (Dict[str, Any]): Metrics output from compute_har_metrics.
        filepath (Union[str, Path]): Destination file path.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def metrics_summary_table(results_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """Construct a clean comparison DataFrame across multiple experimental runs.

    Args:
        results_list (List[Dict[str, Any]]): List of dicts, each with an 'experiment_name'
            or 'model_name' key and scalar metric values.

    Returns:
        pd.DataFrame: Formatted comparison table.
    """
    rows = []
    for r in results_list:
        row = {
            "Model / Experiment": r.get("model_name", r.get("experiment_name", "Unknown")),
            "Accuracy": f"{r.get('accuracy', 0.0) * 100:.2f}%",
            "Macro F1": f"{r.get('macro_f1', 0.0) * 100:.2f}%",
            "Weighted F1": f"{r.get('weighted_f1', 0.0) * 100:.2f}%",
            "Macro Precision": f"{r.get('macro_precision', 0.0) * 100:.2f}%",
            "Macro Recall": f"{r.get('macro_recall', 0.0) * 100:.2f}%",
        }
        rows.append(row)
    return pd.DataFrame(rows)
