"""Automated Body-Region and Sensor Ablation Study for OPPORTUNITY HAR.

Evaluates how removing specific anatomical body-region sensor groups impacts
recognition performance compared to the full-body sensor baseline.

Conditions:
    1. All on-body sensors (133 channels)
    2. Without Left-Arm sensors
    3. Without Right-Arm sensors
    4. Without Trunk sensors
    5. Without Left-Leg sensors
    6. Without Right-Leg sensors

Usage:
    python -m src.experiments.run_ablation --data_path data/processed/opportunity_locomotion_subset.npz
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
from src.evaluation.metrics import metrics_summary_table, save_metrics_to_json
from src.visualization.plots import plot_confusion_matrix

logger = get_logger("run_ablation")


def run_ablation_study(
    data_path: Path,
    output_dir: Path,
) -> pd.DataFrame:
    """Execute complete body-region ablation experiments on CPU."""
    logger.info("Loading preprocessed dataset from: %s", data_path)
    npz = np.load(data_path, allow_pickle=True)

    X_train_raw = npz["X_train"]  # (N, 133, T)
    y_train = npz["y_train"]
    X_test_raw = npz["X_test"]
    y_test = npz["y_test"]

    class_names_dict = json.loads(str(npz["class_names"]))
    class_names = [class_names_dict.get(str(i), class_names_dict.get(i, f"C{i}")) for i in range(len(class_names_dict))]

    # Map raw 133 channels to 0-indexed positions within X_train_raw
    sorted_all_cols = get_on_body_column_indices()
    col_to_idx = {col: idx for idx, col in enumerate(sorted_all_cols)}

    # Define ablation scenarios
    ablation_scenarios: Dict[str, List[int]] = {
        "All Sensors (133 ch)": list(range(len(sorted_all_cols))),
    }

    for region in BODY_REGIONS:
        dropped_cols = set(ANATOMICAL_SENSOR_COLUMNS[region])
        kept_indices = [col_to_idx[col] for col in sorted_all_cols if col not in dropped_cols]
        ablation_scenarios[f"Without {region.replace('_', ' ').title()}"] = kept_indices

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    full_payload = {}

    for name, channel_indices in ablation_scenarios.items():
        logger.info("Running condition: '%s' (%d channels)...", name, len(channel_indices))
        # Select active channels
        X_tr_sub = X_train_raw[:, channel_indices, :]
        X_te_sub = X_test_raw[:, channel_indices, :]

        # Extract temporal statistics
        feats_train = extract_statistical_features(X_tr_sub)
        feats_test = extract_statistical_features(X_te_sub)

        # Train Random Forest
        _, metrics, _ = train_random_forest(
            feats_train, y_train, feats_test, y_test,
            n_estimators=100, random_state=42
        )

        acc = metrics["accuracy"]
        f1 = metrics["macro_f1"]
        logger.info("  %s -> Accuracy: %.2f%%, Macro F1: %.2f%%", name, acc * 100, f1 * 100)

        summary_rows.append({
            "Ablation Condition": name,
            "Channels": len(channel_indices),
            "Accuracy": f"{acc * 100:.2f}%",
            "Macro F1": f"{f1 * 100:.2f}%",
            "Weighted F1": f"{metrics['weighted_f1'] * 100:.2f}%",
            "Macro Precision": f"{metrics['macro_precision'] * 100:.2f}%",
            "Macro Recall": f"{metrics['macro_recall'] * 100:.2f}%",
            "raw_macro_f1": f1,
            "raw_accuracy": acc,
        })
        full_payload[name] = metrics

        # Save confusion matrix plot
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        plot_confusion_matrix(
            metrics["confusion_matrix"],
            class_names=class_names,
            title=f"Confusion Matrix: {name}",
            save_path=output_dir / f"cm_{safe_name}.png",
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "ablation_summary.csv", index=False)
    save_metrics_to_json(full_payload, output_dir / "results.json")

    # Generate comparative bar plot of Macro F1 across ablations
    fig, ax = plt.subplots(figsize=(10, 5))
    bar_data = summary_df.copy()
    bar_data["Macro_F1_Pct"] = bar_data["raw_macro_f1"] * 100

    sns.barplot(data=bar_data, x="Ablation Condition", y="Macro_F1_Pct", ax=ax, palette="Blues_r")
    ax.set_ylabel("Macro F1 (%)", fontsize=11)
    ax.set_title("Body-Region Ablation Study (Macro F1 on OPPORTUNITY Test Split)", fontsize=12)
    plt.xticks(rotation=25, ha="right")
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.tight_layout()
    fig.savefig(output_dir / "ablation_f1_comparison.png", dpi=300)
    plt.close(fig)

    logger.info("Saved ablation results and comparison plot to: %s", output_dir)
    return summary_df


def main():
    parser = argparse.ArgumentParser(description="Run body-region ablation on OPPORTUNITY")
    parser.add_argument("--data_path", type=str, default="data/processed/opportunity_locomotion_subset.npz")
    parser.add_argument("--output_dir", type=str, default="results/body_ablation")
    args = parser.parse_args()

    df = run_ablation_study(Path(args.data_path), Path(args.output_dir))
    print("\n" + "=" * 75)
    print(" OPPORTUNITY Body-Region Ablation Study Results")
    print("=" * 75)
    print(df[["Ablation Condition", "Channels", "Accuracy", "Macro F1", "Weighted F1"]].to_string(index=False))
    print("=" * 75)


if __name__ == "__main__":
    main()
