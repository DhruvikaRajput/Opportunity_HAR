"""Leave-One-Subject-Out (LOSO) Cross-Subject Cross-Validation for OPPORTUNITY HAR.

Iterates over all 4 subjects (S1, S2, S3, S4):
    In fold i: Train on 3 subjects, Evaluate on held-out subject i.
Calculates and records:
    - Per-subject Accuracy, Precision, Recall, Macro F1, Weighted F1
    - Aggregated Mean and Standard Deviation across subjects.

COMPUTE SAFETY RULE:
    - Classical models (Random Forest) execute on CPU.
    - Neural models (CNN, Fixed GNN, Adaptive GNN) require CUDA. Refuses full neural training on laptop CPU.
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
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.logging import get_logger
from src.data.opportunity_loader import OpportunityLoader
from src.data.preprocessing import SensorStandardScaler
from src.data.windowing import create_sliding_windows
from src.data.preprocess_opportunity import remap_labels_to_continuous_indices
from src.models.classical import extract_statistical_features, train_random_forest
from src.evaluation.metrics import compute_har_metrics

logger = get_logger("run_loso")

ALL_SUBJECTS = ["S1", "S2", "S3", "S4"]
RUNS = ["ADL1", "ADL2", "ADL3", "ADL4", "ADL5", "Drill"]


def run_classical_loso(
    output_dir: Path,
    max_runs_per_sub: int = 3,  # keep CPU runtime reasonable
) -> pd.DataFrame:
    """Execute Leave-One-Subject-Out cross-validation for Random Forest on CPU."""
    loader = OpportunityLoader(target_track="Locomotion", sensor_selection="on_body")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading continuous recordings for LOSO subjects: %s...", ALL_SUBJECTS)
    subject_data: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}

    for sub in ALL_SUBJECTS:
        subject_data[sub] = []
        for r in RUNS[:max_runs_per_sub]:
            try:
                data, labels, _ = loader.load_recording(sub, r, interpolate_nans=True)
                subject_data[sub].append((data, labels))
            except FileNotFoundError:
                continue

    per_subject_results = []

    for test_sub in ALL_SUBJECTS:
        train_subs = [s for s in ALL_SUBJECTS if s != test_sub]
        logger.info("LOSO Fold: Train on %s -> Test on held-out %s", train_subs, test_sub)

        # 1. Stack training raw data to fit normalizer (strictly no leakage from test_sub)
        train_raw_list = []
        for s in train_subs:
            for d, _ in subject_data[s]:
                train_raw_list.append(d)

        scaler = SensorStandardScaler()
        scaler.fit(np.concatenate(train_raw_list, axis=0))

        # 2. Window and extract features for training subjects
        train_windows = []
        train_labels = []
        for s in train_subs:
            for d, l in subject_data[s]:
                norm_d = scaler.transform(d)
                w, w_lbl = create_sliding_windows(norm_d, l, window_size=30, stride=15, channel_first=True)
                if len(w) > 0:
                    train_windows.append(w)
                    train_labels.append(w_lbl)

        X_tr = np.concatenate(train_windows, axis=0)
        y_tr_raw = np.concatenate(train_labels, axis=0)

        # 3. Window and extract features for held-out test subject
        test_windows = []
        test_labels = []
        for d, l in subject_data[test_sub]:
            norm_d = scaler.transform(d)
            w, w_lbl = create_sliding_windows(norm_d, l, window_size=30, stride=15, channel_first=True)
            if len(w) > 0:
                test_windows.append(w)
                test_labels.append(w_lbl)

        X_te = np.concatenate(test_windows, axis=0)
        y_te_raw = np.concatenate(test_labels, axis=0)

        # Remap labels
        y_tr, class_map, _ = remap_labels_to_continuous_indices(y_tr_raw, track="Locomotion")
        y_te = np.array([class_map.get(val, 0) for val in y_te_raw], dtype=np.int64)

        # Extract features
        feats_tr = extract_statistical_features(X_tr)
        feats_te = extract_statistical_features(X_te)

        # Train Random Forest
        _, metrics, _ = train_random_forest(feats_tr, y_tr, feats_te, y_te, n_estimators=80, random_state=42)

        logger.info("  Held-out Subject %s -> Accuracy: %.2f%%, Macro F1: %.2f%%", test_sub, metrics["accuracy"] * 100, metrics["macro_f1"] * 100)

        per_subject_results.append({
            "Held-Out Subject": test_sub,
            "Accuracy": metrics["accuracy"],
            "Macro F1": metrics["macro_f1"],
            "Weighted F1": metrics["weighted_f1"],
            "Macro Precision": metrics["macro_precision"],
            "Macro Recall": metrics["macro_recall"],
        })

    df = pd.DataFrame(per_subject_results)

    # Compute mean and standard deviation
    mean_row = {"Held-Out Subject": "Mean"}
    std_row = {"Held-Out Subject": "Std Dev"}

    for metric in ["Accuracy", "Macro F1", "Weighted F1", "Macro Precision", "Macro Recall"]:
        mean_row[metric] = df[metric].mean()
        std_row[metric] = df[metric].std()

    summary_df = pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    summary_df.to_csv(output_dir / "loso_random_forest_results.csv", index=False)

    # Formatting for display
    display_df = summary_df.copy()
    for col in ["Accuracy", "Macro F1", "Weighted F1", "Macro Precision", "Macro Recall"]:
        display_df[col] = display_df[col].apply(lambda x: f"{x * 100:.2f}%")

    # Plot per-subject Macro F1
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=df, x="Held-Out Subject", y="Macro F1", ax=ax, palette="Blues")
    ax.axhline(mean_row["Macro F1"], color="red", linestyle="--", label=f"Mean: {mean_row['Macro F1']*100:.2f}%")
    ax.set_ylabel("Macro F1 Score")
    ax.set_title("Leave-One-Subject-Out (LOSO) Cross-Validation: Random Forest", fontsize=12)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "loso_random_forest_f1.png", dpi=300)
    plt.close(fig)

    return display_df


def main():
    parser = argparse.ArgumentParser(description="Run LOSO cross-validation on OPPORTUNITY")
    parser.add_argument("--model", type=str, default="random_forest", choices=["random_forest", "cnn", "fixed_gnn", "adaptive_gnn"])
    parser.add_argument("--output_dir", type=str, default="results/loso")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)

    if args.model in ["cnn", "fixed_gnn", "adaptive_gnn"]:
        if not torch.cuda.is_available() and not args.smoke_test:
            print("\n[STOPPED - COMPUTE RULE ENFORCED]")
            print(f"Neural model '{args.model}' LOSO requires CUDA GPU.")
            print("Full neural LOSO cross-validation must run in Google Colab (notebooks/02_gpu_training.ipynb).")
            print("To verify classical ML LOSO on CPU, run: python -m src.experiments.run_loso --model random_forest\n")
            sys.exit(0)

    print("\nRunning LOSO cross-validation for Random Forest on CPU...")
    df = run_classical_loso(out_dir, max_runs_per_sub=2)
    print("\n" + "=" * 75)
    print(" OPPORTUNITY Leave-One-Subject-Out (LOSO) Results: Random Forest")
    print("=" * 75)
    print(df.to_string(index=False))
    print("=" * 75)


if __name__ == "__main__":
    main()
