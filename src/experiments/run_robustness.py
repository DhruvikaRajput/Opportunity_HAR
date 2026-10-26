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
    n_repeats: int = 1,
) -> pd.DataFrame:
    """Run controlled sensor loss experiments on CPU using Random Forest.

    Args:
        data_path (Path): Path to .npz preprocessed dataset archive.
        output_dir (Path): Destination directory for CSV and plots.
        n_repeats (int): Number of repeated random dropout masks to sample per rate
            (default 1 for canonical benchmark; >1 for multi-trial confidence estimation).
    """
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
    C = X_test_raw.shape[1]
    total_channels = C

    # Free raw 3D window tensors to conserve memory
    del X_train_raw, X_test_raw
    import gc
    gc.collect()

    clf, clean_metrics, _ = train_random_forest(X_tr_feats, y_train, X_te_feats_clean, y_test, random_state=42)
    clean_f1 = clean_metrics["macro_f1"]
    clean_acc = clean_metrics["accuracy"]

    logger.info("Clean Baseline -> Accuracy: %.2f%%, Macro F1: %.2f%%", clean_acc * 100, clean_f1 * 100)

    output_dir.mkdir(parents=True, exist_ok=True)
    robustness_records = []
    full_payload = {"clean_baseline": clean_metrics}

    # Scenario A: Random Sensor Dropout (10%, 25%, 50%, 75%)
    dropout_rates = [0.0, 0.10, 0.25, 0.50, 0.75]
    canonical_rng = np.random.RandomState(42)

    for rate in dropout_rates:
        if rate == 0.0:
            rec = {
                "Failure Scenario": "0% Failure (Clean)",
                "Condition Type": "Random Dropout",
                "Dropped Channels": 0,
                "Accuracy": f"{clean_acc * 100:.2f}%",
                "Macro F1": f"{clean_f1 * 100:.2f}%",
                "Degradation (Macro F1)": "0.00%",
                "raw_macro_f1": clean_f1,
            }
            if n_repeats > 1:
                rec["Macro F1 (Mean±Std)"] = f"{clean_f1 * 100:.2f}% ± 0.00%"
                rec["Degradation (Mean±Std)"] = "0.00% ± 0.00%"
            robustness_records.append(rec)
            continue

        num_drop = int(total_channels * rate)
        trial_metrics = []

        for rep in range(n_repeats):
            if rep == 0:
                drop_indices = canonical_rng.choice(total_channels, size=num_drop, replace=False)
            else:
                rng = np.random.RandomState(100 * rep + int(rate * 100))
                drop_indices = rng.choice(total_channels, size=num_drop, replace=False)

            # Mask corrupted feature columns (mean, std, min, max) to 0.0
            feats_corrupted = X_te_feats_clean.copy()
            corrupt_cols = (
                list(drop_indices)
                + [C + c for c in drop_indices]
                + [2 * C + c for c in drop_indices]
                + [3 * C + c for c in drop_indices]
            )
            feats_corrupted[:, corrupt_cols] = 0.0

            preds = clf.predict(feats_corrupted)
            m = compute_har_metrics(y_test, preds)
            trial_metrics.append(m)

        # Primary benchmark metrics (first draw, seed 42)
        base_m = trial_metrics[0]
        deg_f1 = (clean_f1 - base_m["macro_f1"]) * 100
        deg_acc = (clean_acc - base_m["accuracy"]) * 100

        rec = {
            "Failure Scenario": f"{int(rate*100)}% Random Sensor Failure",
            "Condition Type": "Random Dropout",
            "Dropped Channels": num_drop,
            "Accuracy": f"{base_m['accuracy'] * 100:.2f}%",
            "Macro F1": f"{base_m['macro_f1'] * 100:.2f}%",
            "Degradation (Macro F1)": f"-{deg_f1:.2f}%",
            "raw_macro_f1": base_m["macro_f1"],
        }

        if n_repeats > 1:
            all_f1s = [m["macro_f1"] * 100 for m in trial_metrics]
            all_degs = [(clean_f1 - m["macro_f1"]) * 100 for m in trial_metrics]
            rec["Macro F1 (Mean±Std)"] = f"{np.mean(all_f1s):.2f}% ± {np.std(all_f1s):.2f}%"
            rec["Degradation (Mean±Std)"] = f"-{np.mean(all_degs):.2f}% ± {np.std(all_degs):.2f}%"
            logger.info("    Dropout %d%% (%d trials) -> F1: %s, Degradation: %s",
                        int(rate*100), n_repeats, rec["Macro F1 (Mean±Std)"], rec["Degradation (Mean±Std)"])

        robustness_records.append(rec)
        full_payload[f"dropout_{int(rate*100)}pct"] = base_m if n_repeats == 1 else {"primary": base_m, "trials": trial_metrics}

    # Scenario B: Anatomical Body Region Failure (complete regional outage)
    sorted_cols = get_on_body_column_indices()
    col_to_idx = {col: idx for idx, col in enumerate(sorted_cols)}

    for region in BODY_REGIONS:
        reg_cols = ANATOMICAL_SENSOR_COLUMNS[region]
        mask_indices = [col_to_idx[c] for c in reg_cols]

        feats_corrupted = X_te_feats_clean.copy()
        corrupt_cols = (
            list(mask_indices)
            + [C + c for c in mask_indices]
            + [2 * C + c for c in mask_indices]
            + [3 * C + c for c in mask_indices]
        )
        feats_corrupted[:, corrupt_cols] = 0.0

        preds = clf.predict(feats_corrupted)
        metrics = compute_har_metrics(y_test, preds)

        deg_f1 = (clean_f1 - metrics["macro_f1"]) * 100
        rec = {
            "Failure Scenario": f"Complete {region.replace('_', ' ').title()} Outage",
            "Condition Type": "Regional Failure",
            "Dropped Channels": len(mask_indices),
            "Accuracy": f"{metrics['accuracy'] * 100:.2f}%",
            "Macro F1": f"{metrics['macro_f1'] * 100:.2f}%",
            "Degradation (Macro F1)": f"-{deg_f1:.2f}%",
            "raw_macro_f1": metrics["macro_f1"],
        }
        if n_repeats > 1:
            rec["Macro F1 (Mean±Std)"] = f"{metrics['macro_f1'] * 100:.2f}% (deterministic)"
            rec["Degradation (Mean±Std)"] = f"-{deg_f1:.2f}% (deterministic)"
        robustness_records.append(rec)
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


run_robustness_study = run_robustness_experiments


def main():
    parser = argparse.ArgumentParser(description="Run sensor robustness experiments on OPPORTUNITY")
    parser.add_argument("--data_path", type=str, default="data/processed/opportunity_locomotion_subset.npz")
    parser.add_argument("--output_dir", type=str, default="results/robustness")
    parser.add_argument("--n_repeats", type=int, default=1, help="Number of repeated random masks per dropout level")
    args = parser.parse_args()

    df = run_robustness_experiments(Path(args.data_path), Path(args.output_dir), n_repeats=args.n_repeats)
    print("\n" + "=" * 75)
    print(" OPPORTUNITY Sensor-Failure Robustness Study Results")
    print("=" * 75)
    disp_cols = ["Failure Scenario", "Dropped Channels", "Accuracy", "Macro F1", "Degradation (Macro F1)"]
    if "Degradation (Mean±Std)" in df.columns:
        disp_cols.append("Degradation (Mean±Std)")
    print(df[disp_cols].to_string(index=False))
    print("=" * 75)


if __name__ == "__main__":
    main()
