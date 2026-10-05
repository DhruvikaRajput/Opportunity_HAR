"""Sensor failure and missing-modality robustness experiments for OPPORTUNITY HAR.

Simulates sensor dropout / failure during evaluation:
    1. Random Sensor Dropout (10%, 25%, 50%, 75% of channels masked to 0)
    2. Anatomical Region Failure (complete loss of Left Arm, Right Arm, Trunk, or Legs)

Measures and plots performance degradation relative to the full-sensor benchmark.
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.logging import get_logger
from src.features.body_regions import (
    BODY_REGIONS,
    ANATOMICAL_SENSOR_COLUMNS,
    get_on_body_column_indices,
)
from src.models.classical import extract_statistical_features, train_random_forest
from src.evaluation.metrics import compute_har_metrics, save_metrics_to_json

logger = get_logger("run_robustness")


def run_robustness_experiments(
    data_path: Path,
    output_dir: Path,
) -> pd.DataFrame:
    """Run controlled sensor loss experiments on CPU using Random Forest."""
    logger.info("Loading preprocessed dataset from: %s", data_path)
    npz = np.load(data_path, allow_pickle=True)

    X_train_raw = npz["X_train"]  # (N, 133, T)
    y_train = npz["y_train"]
    X_test_raw = npz["X_test"]
    y_test = npz["y_test"]

    # 1. Train full-sensor baseline model on clean training data
    logger.info("Training clean reference Random Forest on all 133 channels...")
    X_tr_feats = extract_statistical_features(X_train_raw)
    X_te_feats_clean = extract_statistical_features(X_test_raw)

    clf, clean_metrics, _ = train_random_forest(X_tr_feats, y_train, X_te_feats_clean, y_test, random_state=42)
    clean_f1 = clean_metrics["macro_f1"]
    clean_acc = clean_metrics["accuracy"]

    logger.info("Clean Baseline -> Accuracy: %.2f%%, Macro F1: %.2f%%", clean_acc * 100, clean_f1 * 100)

    output_dir.mkdir(parents=True, exist_ok=True)
    robustness_records = []
    full_payload = {"clean_baseline": clean_metrics}

    # Scenario A: Random Sensor Dropout (10%, 25%, 50%, 75%)
    dropout_rates = [0.0, 0.10, 0.25, 0.50, 0.75]
    total_channels = X_test_raw.shape[1]
    rng = np.random.RandomState(42)

    for rate in dropout_rates:
        if rate == 0.0:
            robustness_records.append({
                "Failure Scenario": "0% Failure (Clean)",
                "Condition Type": "Random Dropout",
                "Dropped Channels": 0,
                "Accuracy": f"{clean_acc * 100:.2f}%",
                "Macro F1": f"{clean_f1 * 100:.2f}%",
                "Degradation (Macro F1)": "0.00%",
                "raw_macro_f1": clean_f1,
            })
            continue

        num_drop = int(total_channels * rate)
        drop_indices = rng.choice(total_channels, size=num_drop, replace=False)

        # Mask corrupted channels to 0 in test windows
        X_te_corrupted = X_test_raw.copy()
        X_te_corrupted[:, drop_indices, :] = 0.0

        feats_corrupted = extract_statistical_features(X_te_corrupted)
        preds = clf.predict(feats_corrupted)
        metrics = compute_har_metrics(y_test, preds)

        deg = (clean_f1 - metrics["macro_f1"]) * 100
        robustness_records.append({
            "Failure Scenario": f"{int(rate*100)}% Random Sensor Failure",
            "Condition Type": "Random Dropout",
            "Dropped Channels": num_drop,
            "Accuracy": f"{metrics['accuracy'] * 100:.2f}%",
            "Macro F1": f"{metrics['macro_f1'] * 100:.2f}%",
            "Degradation (Macro F1)": f"-{deg:.2f}%",
            "raw_macro_f1": metrics["macro_f1"],
        })
        full_payload[f"dropout_{int(rate*100)}pct"] = metrics

    # Scenario B: Anatomical Body Region Failure (complete regional outage)
    sorted_cols = get_on_body_column_indices()
    col_to_idx = {col: idx for idx, col in enumerate(sorted_cols)}

    for region in BODY_REGIONS:
        reg_cols = ANATOMICAL_SENSOR_COLUMNS[region]
        mask_indices = [col_to_idx[c] for c in reg_cols]

        X_te_corrupted = X_test_raw.copy()
        X_te_corrupted[:, mask_indices, :] = 0.0

        feats_corrupted = extract_statistical_features(X_te_corrupted)
        preds = clf.predict(feats_corrupted)
        metrics = compute_har_metrics(y_test, preds)

        deg = (clean_f1 - metrics["macro_f1"]) * 100
        robustness_records.append({
            "Failure Scenario": f"Complete {region.replace('_', ' ').title()} Outage",
            "Condition Type": "Regional Failure",
            "Dropped Channels": len(mask_indices),
            "Accuracy": f"{metrics['accuracy'] * 100:.2f}%",
            "Macro F1": f"{metrics['macro_f1'] * 100:.2f}%",
            "Degradation (Macro F1)": f"-{deg:.2f}%",
            "raw_macro_f1": metrics["macro_f1"],
        })
        full_payload[f"outage_{region}"] = metrics

    df = pd.DataFrame(robustness_records)
    df.to_csv(output_dir / "robustness_summary.csv", index=False)
    save_metrics_to_json(full_payload, output_dir / "results.json")

    # Generate Robustness Degradation Curve Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    rand_data = df[df["Condition Type"] == "Random Dropout"].copy()
    drop_pcts = [0, 10, 25, 50, 75]
    f1_vals = rand_data["raw_macro_f1"].values * 100

    ax.plot(drop_pcts, f1_vals, marker="o", color="crimson", linewidth=2.0, label="Random Sensor Dropout")
    ax.axhline(clean_f1 * 100, color="gray", linestyle="--", label=f"Clean Performance: {clean_f1*100:.2f}%")
    ax.set_xlabel("Percentage of Sensors Failed (%)", fontsize=10)
    ax.set_ylabel("Macro F1 (%)", fontsize=10)
    ax.set_title("OPPORTUNITY Sensor-Failure Robustness Degradation Curve", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "robustness_curves.png", dpi=300)
    plt.close(fig)

    logger.info("Saved robustness analysis and degradation curves to: %s", output_dir)
    return df


def main():
    parser = argparse.ArgumentParser(description="Run sensor robustness experiments on OPPORTUNITY")
    parser.add_argument("--data_path", type=str, default="data/processed/opportunity_locomotion_subset.npz")
    parser.add_argument("--output_dir", type=str, default="results/robustness")
    args = parser.parse_args()

    df = run_robustness_experiments(Path(args.data_path), Path(args.output_dir))
    print("\n" + "=" * 75)
    print(" OPPORTUNITY Sensor-Failure Robustness Study Results")
    print("=" * 75)
    print(df[["Failure Scenario", "Dropped Channels", "Accuracy", "Macro F1", "Degradation (Macro F1)"]].to_string(index=False))
    print("=" * 75)


if __name__ == "__main__":
    main()
