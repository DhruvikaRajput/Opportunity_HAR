"""Execute and evaluate CPU-based classical machine learning baselines for OPPORTUNITY HAR.

Usage:
    python -m src.experiments.run_baseline --data_path data/processed/opportunity_locomotion_subset.npz
"""

import sys
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.logging import get_logger
from src.models.classical import (
    extract_statistical_features,
    train_random_forest,
    train_logistic_regression,
    train_linear_svm,
)
from src.evaluation.metrics import metrics_summary_table, save_metrics_to_json
from src.visualization.plots import plot_confusion_matrix

logger = get_logger("run_baseline")


def run_baseline_experiments(data_path: Path, output_dir: Path) -> pd.DataFrame:
    """Run classical ML baselines (Random Forest, Logistic Regression, Linear SVM) on CPU."""
    logger.info("Loading preprocessed dataset from: %s", data_path)
    npz = np.load(data_path, allow_pickle=True)

    X_train_raw = npz["X_train"]
    y_train = npz["y_train"]
    X_test_raw = npz["X_test"]
    y_test = npz["y_test"]

    class_names_dict = json.loads(str(npz["class_names"]))
    class_names = [class_names_dict.get(str(i), class_names_dict.get(i, f"C{i}")) for i in range(len(class_names_dict))]

    logger.info("Extracting summary statistical features per channel across windows...")
    X_train = extract_statistical_features(X_train_raw)
    X_test = extract_statistical_features(X_test_raw)
    logger.info("Feature matrix shape: Train=%s, Test=%s", X_train.shape, X_test.shape)

    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []
    full_payload = {}

    # 1. Random Forest
    logger.info("Training Random Forest baseline on CPU...")
    _, rf_metrics, rf_preds = train_random_forest(X_train, y_train, X_test, y_test, n_estimators=100)
    logger.info("  Random Forest -> Accuracy: %.2f%%, Macro F1: %.2f%%", rf_metrics["accuracy"] * 100, rf_metrics["macro_f1"] * 100)
    all_results.append({"model_name": "Random Forest", **rf_metrics})
    full_payload["Random_Forest"] = rf_metrics

    plot_confusion_matrix(
        rf_metrics["confusion_matrix"],
        class_names=class_names,
        title="Random Forest Test Confusion Matrix",
        save_path=output_dir / "random_forest_cm.png",
    )

    # 2. Logistic Regression
    logger.info("Training Logistic Regression baseline on CPU...")
    _, lr_metrics, lr_preds = train_logistic_regression(X_train, y_train, X_test, y_test, max_iter=300)
    logger.info("  Logistic Regression -> Accuracy: %.2f%%, Macro F1: %.2f%%", lr_metrics["accuracy"] * 100, lr_metrics["macro_f1"] * 100)
    all_results.append({"model_name": "Logistic Regression", **lr_metrics})
    full_payload["Logistic_Regression"] = lr_metrics

    plot_confusion_matrix(
        lr_metrics["confusion_matrix"],
        class_names=class_names,
        title="Logistic Regression Test Confusion Matrix",
        save_path=output_dir / "logistic_regression_cm.png",
    )

    # 3. Linear SVM
    logger.info("Training Linear SVM baseline on CPU...")
    _, svm_metrics, svm_preds = train_linear_svm(X_train, y_train, X_test, y_test, max_iter=500)
    logger.info("  Linear SVM -> Accuracy: %.2f%%, Macro F1: %.2f%%", svm_metrics["accuracy"] * 100, svm_metrics["macro_f1"] * 100)
    all_results.append({"model_name": "Linear SVM", **svm_metrics})
    full_payload["Linear_SVM"] = svm_metrics

    plot_confusion_matrix(
        svm_metrics["confusion_matrix"],
        class_names=class_names,
        title="Linear SVM Test Confusion Matrix",
        save_path=output_dir / "linear_svm_cm.png",
    )

    # Save summary table and metrics JSON
    summary_df = metrics_summary_table(all_results)
    summary_df.to_csv(output_dir / "baseline_comparison.csv", index=False)
    save_metrics_to_json(full_payload, output_dir / "results.json")

    logger.info("Saved all classical baseline results to: %s", output_dir)
    return summary_df


def main():
    parser = argparse.ArgumentParser(description="Run classical ML baselines on OPPORTUNITY dataset")
    parser.add_argument("--data_path", type=str, default="data/processed/opportunity_locomotion_subset.npz")
    parser.add_argument("--output_dir", type=str, default="results/baseline")
    args = parser.parse_args()

    data_file = Path(args.data_path)
    out_dir = Path(args.output_dir)
    df = run_baseline_experiments(data_file, out_dir)
    print("\n" + "=" * 70)
    print(" OPPORTUNITY Classical ML Baseline Results")
    print("=" * 70)
    print(df.to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    main()
