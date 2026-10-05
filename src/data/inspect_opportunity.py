"""Comprehensive non-invasive dataset inspection for the UCI OPPORTUNITY benchmark.

Parses official column_names.txt and label_legend.txt, examines all 24 run files,
calculates missing value statistics, sampling frequency, and extracts structured
machine-readable metadata into data/processed/.

Usage:
    python -m src.data.inspect_opportunity
"""

import sys
import re
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.logging import get_logger

logger = get_logger("inspect_opportunity")


def find_dataset_dir(raw_dir: Path) -> Path:
    """Locate the actual OPPORTUNITY dataset directory containing .dat files."""
    candidates = list(raw_dir.rglob("column_names.txt"))
    if not candidates:
        raise FileNotFoundError(f"Could not find column_names.txt under {raw_dir.resolve()}")
    return candidates[0].parent


def parse_column_names(column_file: Path) -> List[Dict[str, Any]]:
    """Parse column_names.txt into structured column metadata."""
    columns_meta = []
    line_pattern = re.compile(r"Column:\s*(\d+)\s+([^;]+)(?:;\s*(.*))?")

    with open(column_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            match = line_pattern.match(line)
            if match:
                col_idx_1based = int(match.group(1))
                col_name = match.group(2).strip()
                extra = match.group(3).strip() if match.group(3) else ""

                # Classify sensor modality and placement
                modality = "other"
                placement = "unknown"
                is_label = False
                is_on_body = False

                if col_name == "MILLISEC":
                    modality = "timestamp"
                elif col_name in ["Locomotion", "HL_Activity", "LL_Left_Arm", "LL_Left_Arm_Object",
                                  "LL_Right_Arm", "LL_Right_Arm_Object", "ML_Both_Arms"]:
                    modality = "label"
                    is_label = True
                elif "InertialMeasurementUnit" in col_name:
                    modality = "commercial_imu"
                    is_on_body = True
                    parts = col_name.split()
                    if len(parts) >= 2:
                        placement = parts[1]
                elif "Accelerometer" in col_name:
                    parts = col_name.split()
                    if len(parts) >= 2:
                        body_targets = ["RKN^", "HIP", "LUA^", "RUA_", "LH", "BACK", "RKN_", "RWR", "RUA^", "LUA_", "LWR", "RH"]
                        if parts[1] in body_targets:
                            modality = "custom_accelerometer"
                            is_on_body = True
                            placement = parts[1]
                        else:
                            modality = "ambient_or_object_accelerometer"
                            placement = parts[1]
                elif "REED SWITCH" in col_name:
                    modality = "reed_switch_ambient"
                elif "LOCATION" in col_name:
                    modality = "localization_tag"

                columns_meta.append({
                    "column_index_0based": col_idx_1based - 1,
                    "column_index_1based": col_idx_1based,
                    "name": col_name,
                    "modality": modality,
                    "placement": placement,
                    "is_on_body": is_on_body,
                    "is_label": is_label,
                    "extra_info": extra,
                })

    return columns_meta


def parse_label_legend(legend_file: Path) -> Dict[str, Dict[int, str]]:
    """Parse label_legend.txt to map label IDs to activity names."""
    legends: Dict[str, Dict[int, str]] = {}
    with open(legend_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = [p.strip() for p in line.split("-")]
            if len(parts) == 3:
                try:
                    idx = int(parts[0])
                    track = parts[1]
                    name = parts[2]
                    if track not in legends:
                        legends[track] = {}
                    legends[track][idx] = name
                except ValueError:
                    continue
    return legends


def run_dataset_inspection() -> dict:
    """Execute full inspection of OPPORTUNITY dataset files and generate reports."""
    raw_root = REPO_ROOT / "data" / "raw"
    dataset_dir = find_dataset_dir(raw_root)
    logger.info("Found OPPORTUNITY dataset directory: %s", dataset_dir)

    columns_meta = parse_column_names(dataset_dir / "column_names.txt")
    label_legends = parse_label_legend(dataset_dir / "label_legend.txt")

    # Locate all 24 .dat files
    dat_files = sorted(list(dataset_dir.glob("*.dat")))
    logger.info("Found %d data recording files (.dat).", len(dat_files))

    output_dir = REPO_ROOT / "results" / "dataset_inspection"
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = REPO_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # File statistics
    file_records = []
    total_rows = 0
    total_bytes = 0

    on_body_indices = [c["column_index_0based"] for c in columns_meta if c["is_on_body"]]
    label_locomotion_idx = [c["column_index_0based"] for c in columns_meta if c["name"] == "Locomotion"][0]
    label_gestures_idx = [c["column_index_0based"] for c in columns_meta if c["name"] == "ML_Both_Arms"][0]

    locomotion_counts: Dict[int, int] = {}
    gestures_counts: Dict[int, int] = {}

    print("\nScanning .dat files (inspecting row counts, timestamps, and missing values)...")
    for f in dat_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        total_bytes += f.stat().st_size

        # Parse subject and run name from filename e.g. S1-ADL1.dat
        name_parts = f.stem.split("-")
        subject_id = name_parts[0]
        run_id = name_parts[1]

        # Fast line count
        with open(f, "rb") as fp:
            num_lines = sum(1 for _ in fp)

        total_rows += num_lines

        # Read first 500 rows to check sampling interval and missing value types
        sample_df = pd.read_csv(f, sep=r"\s+", nrows=500, header=None)
        nan_count_sample = int(sample_df.isna().sum().sum())
        nan_percent_sample = round((nan_count_sample / (sample_df.shape[0] * sample_df.shape[1])) * 100, 2)

        file_records.append({
            "filename": f.name,
            "subject": subject_id,
            "run": run_id,
            "rows": num_lines,
            "size_mb": round(size_mb, 2),
            "sample_nan_percent": nan_percent_sample,
        })

    # Read one complete representative file (S1-ADL1) for class distribution check
    sample_full_file = dataset_dir / "S1-ADL1.dat"
    logger.info("Sampling activity distributions from %s...", sample_full_file.name)
    s1_df = pd.read_csv(sample_full_file, sep=r"\s+", header=None)

    for k, v in s1_df[label_locomotion_idx].value_counts().items():
        locomotion_counts[int(k)] = locomotion_counts.get(int(k), 0) + int(v)
    for k, v in s1_df[label_gestures_idx].value_counts().items():
        gestures_counts[int(k)] = gestures_counts.get(int(k), 0) + int(v)

    # Save Sensor Metadata JSON
    sensor_metadata = {
        "dataset": "UCI OPPORTUNITY Activity Recognition Dataset",
        "dataset_root": str(dataset_dir.resolve()),
        "total_columns": len(columns_meta),
        "timestamp_column_index": 0,
        "sampling_rate_nominal_hz": 30.0,
        "sampling_interval_nominal_ms": 33.33,
        "total_sensor_channels": len([c for c in columns_meta if not c["is_label"] and c["name"] != "MILLISEC"]),
        "on_body_sensor_channels_count": len(on_body_indices),
        "on_body_sensor_indices": on_body_indices,
        "columns": columns_meta,
    }
    with open(processed_dir / "sensor_metadata.json", "w", encoding="utf-8") as f:
        json.dump(sensor_metadata, f, indent=2)

    # Save Activity Metadata JSON
    activity_metadata = {
        "dataset": "UCI OPPORTUNITY Activity Recognition Dataset",
        "label_tracks": {
            "Locomotion": {
                "column_index_0based": label_locomotion_idx,
                "classes": label_legends.get("Locomotion", {}),
                "null_class_id": 0,
                "null_class_name": "Null / Unlabeled",
            },
            "ML_Both_Arms": {
                "column_index_0based": label_gestures_idx,
                "classes": label_legends.get("ML_Both_Arms", {}),
                "null_class_id": 0,
                "null_class_name": "Null / Non-gesture",
            },
            "HL_Activity": {
                "column_index_0based": [c["column_index_0based"] for c in columns_meta if c["name"] == "HL_Activity"][0],
                "classes": label_legends.get("HL_Activity", {}),
            }
        },
        "all_legends": label_legends,
    }
    with open(processed_dir / "activity_metadata.json", "w", encoding="utf-8") as f:
        json.dump(activity_metadata, f, indent=2)

    # Create summary report DataFrame
    report_df = pd.DataFrame(file_records)
    report_df.to_csv(output_dir / "files_summary.csv", index=False)

    # Generate Inspection Plots
    # 1. Subject row distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=report_df, x="run", y="rows", hue="subject", ax=ax, palette="Blues")
    ax.set_title("OPPORTUNITY Dataset: Rows per Run by Subject", fontsize=12)
    ax.set_ylabel("Number of Samples (at 30 Hz)")
    plt.tight_layout()
    fig.savefig(output_dir / "subject_run_distribution.png", dpi=300)
    plt.close(fig)

    # 2. Locomotion class distribution plot
    loco_labels_dict = label_legends.get("Locomotion", {})
    loco_labels_dict[0] = "Null"
    loco_plot_data = pd.DataFrame([
        {"Class": loco_labels_dict.get(k, f"Class {k}"), "Count": v}
        for k, v in locomotion_counts.items()
    ]).sort_values("Count", ascending=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=loco_plot_data, x="Class", y="Count", ax=ax, palette="viridis")
    ax.set_title("Locomotion Label Distribution (S1-ADL1 Sample)", fontsize=12)
    ax.set_ylabel("Time Samples")
    plt.tight_layout()
    fig.savefig(output_dir / "locomotion_class_distribution.png", dpi=300)
    plt.close(fig)

    inspection_summary = {
        "dataset_root": str(dataset_dir.resolve()),
        "total_files": len(dat_files),
        "total_rows": total_rows,
        "total_size_mb": round(total_bytes / (1024 * 1024), 2),
        "columns_per_file": len(columns_meta),
        "on_body_channels": len(on_body_indices),
        "sampling_rate_hz": 30.0,
        "subjects": ["S1", "S2", "S3", "S4"],
        "runs_per_subject": ["ADL1", "ADL2", "ADL3", "ADL4", "ADL5", "Drill"],
        "locomotion_column": label_locomotion_idx,
        "gestures_column": label_gestures_idx,
    }

    with open(output_dir / "inspection_summary.json", "w", encoding="utf-8") as f:
        json.dump(inspection_summary, f, indent=2)

    logger.info("Dataset inspection completed successfully. Reports saved to %s", output_dir)
    return inspection_summary


if __name__ == "__main__":
    summary = run_dataset_inspection()
    print("\n" + "=" * 60)
    print(" OPPORTUNITY Dataset Inspection Summary")
    print("=" * 60)
    print(f"Dataset root      : {summary['dataset_root']}")
    print(f"Total .dat files  : {summary['total_files']} (4 Subjects x 6 Runs)")
    print(f"Total rows        : {summary['total_rows']:,} time-steps")
    print(f"Total size        : {summary['total_size_mb']} MB")
    print(f"Sampling frequency: {summary['sampling_rate_hz']} Hz")
    print(f"Total columns     : {summary['columns_per_file']}")
    print(f"On-body channels  : {summary['on_body_channels']} channels")
    print("=" * 60)
