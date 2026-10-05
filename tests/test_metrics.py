"""Unit tests for HAR evaluation metrics.
"""

import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.metrics import compute_har_metrics, metrics_summary_table


def test_metrics_perfect_predictions():
    """Verify metrics calculation when predictions perfectly match ground truth."""
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 1, 2]

    metrics = compute_har_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["weighted_f1"] == 1.0
    assert np.array(metrics["confusion_matrix"]).diagonal().sum() == len(y_true)


def test_metrics_imbalanced_classes():
    """Verify macro F1 calculation correctly penalizes poor performance on minority class."""
    # 8 samples of class 0, 2 samples of class 1
    y_true = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    # Predict all zeros
    y_pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    metrics = compute_har_metrics(y_true, y_pred)
    # Accuracy is high (80%)
    assert metrics["accuracy"] == 0.8
    # But Macro F1 is significantly lower due to zero score on class 1
    assert metrics["macro_f1"] < 0.5


def test_metrics_summary_table():
    """Verify formatting of the multi-experiment comparison table."""
    results = [
        {"model_name": "CNN_v1", "accuracy": 0.85, "macro_f1": 0.82, "weighted_f1": 0.84, "macro_precision": 0.83, "macro_recall": 0.81},
        {"model_name": "RF_v1", "accuracy": 0.78, "macro_f1": 0.75, "weighted_f1": 0.77, "macro_precision": 0.76, "macro_recall": 0.74},
    ]
    df = metrics_summary_table(results)
    assert len(df) == 2
    assert "Macro F1" in df.columns
    assert "82.00%" in df.loc[0, "Macro F1"]
