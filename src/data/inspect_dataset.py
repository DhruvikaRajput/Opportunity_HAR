"""Dataset inspection script for the UCI OPPORTUNITY benchmark.

Executed as the very first step once raw dataset files are downloaded into data/raw/.
Examines the directory contents, extracts file metadata, examines data structure
(rows, columns, headers, missing values), and reports findings without modifying files.

Usage:
    python -m src.data.inspect_dataset --data_dir data/raw
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.logging import get_logger

logger = get_logger(name="dataset_inspection")


def inspect_raw_directory(data_dir: Path) -> dict:
    """Scan raw dataset directory and report files, sizes, and formats.

    Args:
        data_dir (Path): Path to data/raw directory.

    Returns:
        dict: Inspection report dictionary.
    """
    report = {
        "data_dir": str(data_dir.resolve()),
        "files_found": [],
        "total_files": 0,
        "total_size_mb": 0.0,
    }

    if not data_dir.exists():
        logger.warning("Data directory does not exist: %s", data_dir)
        return report

    all_files = sorted([f for f in data_dir.rglob("*") if f.is_file()])
    report["total_files"] = len(all_files)

    total_bytes = 0
    for f in all_files:
        size = f.stat().st_size
        total_bytes += size
        report["files_found"].append({
            "name": f.name,
            "relative_path": str(f.relative_to(data_dir)),
            "size_kb": round(size / 1024, 2),
            "suffix": f.suffix.lower(),
        })

    report["total_size_mb"] = round(total_bytes / (1024 * 1024), 2)
    return report


def inspect_sample_data_file(file_path: Path, max_rows: int = 1000) -> dict:
    """Inspect structure of a specific raw data file (CSV, DAT, or TXT).

    Args:
        file_path (Path): Path to raw file.
        max_rows (int): Number of rows to read for fast inspection.

    Returns:
        dict: File structure metadata.
    """
    meta = {
        "file_name": file_path.name,
        "rows_inspected": 0,
        "num_columns": 0,
        "sample_columns": [],
        "missing_values_count": 0,
        "dtypes": {},
    }

    try:
        # OPPORTUNITY files often use space-separated or comma-separated values
        # We test space separator first, then comma
        try:
            df = pd.read_csv(file_path, sep=r"\s+", nrows=max_rows, header=None)
        except Exception:
            df = pd.read_csv(file_path, nrows=max_rows)

        meta["rows_inspected"] = len(df)
        meta["num_columns"] = df.shape[1]
        meta["sample_columns"] = list(df.columns[:10])
        meta["missing_values_count"] = int(df.isna().sum().sum())
        meta["dtypes"] = {str(k): str(v) for k, v in df.dtypes.iloc[:10].items()}
    except Exception as e:
        meta["error"] = str(e)

    return meta


def main():
    parser = argparse.ArgumentParser(description="Inspect raw OPPORTUNITY dataset files")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Path to raw data directory")
    args = parser.parse_args()

    raw_path = Path(args.data_dir)
    print("=" * 70)
    print(" UCI OPPORTUNITY Dataset Inspection Report")
    print("=" * 70)

    dir_report = inspect_raw_directory(raw_path)
    print(f"Directory: {dir_report['data_dir']}")
    print(f"Total files found : {dir_report['total_files']}")
    print(f"Total directory size: {dir_report['total_size_mb']} MB")

    if dir_report["total_files"] == 0:
        print("\n [!] data/raw is currently empty or contains only placeholders.")
        print("     Awaiting dataset download completion.")
        print("     Once files are extracted into data/raw, run this script again to inspect.")
    else:
        print("\nFirst 10 files:")
        for file_info in dir_report["files_found"][:10]:
            print(f"  - {file_info['relative_path']} ({file_info['size_kb']} KB)")

        # Inspect first data file if present
        data_exts = {".dat", ".csv", ".txt"}
        candidate_files = [
            raw_path / f["relative_path"]
            for f in dir_report["files_found"]
            if f["suffix"] in data_exts
        ]
        if candidate_files:
            target = candidate_files[0]
            print(f"\nInspecting sample data file: {target.name}")
            sample_meta = inspect_sample_data_file(target)
            print(f"  Columns found : {sample_meta['num_columns']}")
            print(f"  Rows sampled  : {sample_meta['rows_inspected']}")
            print(f"  NaN count     : {sample_meta['missing_values_count']}")

    print("=" * 70)


if __name__ == "__main__":
    main()
