"""Preprocessing and windowing pipeline for the real UCI OPPORTUNITY dataset.

Extracts on-body sensor channels, imputes missing values via linear interpolation,
normalizes sensor channels with statistics fitted STRICTLY on training subjects,
segments continuous streams into sliding windows without cross-run leakage, and
persists train/validation/test tensors into data/processed/.

Usage:
    python -m src.data.preprocess_opportunity --track Locomotion --window_length 30 --stride 15
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.logging import get_logger
from src.data.opportunity_loader import OpportunityLoader
from src.data.preprocessing import SensorStandardScaler
from src.data.windowing import create_sliding_windows
from src.features.body_regions import get_on_body_column_indices

logger = get_logger("preprocess_opportunity")


# Standard OPPORTUNITY Benchmark Splits
DEFAULT_TRAIN_FILES = [
    ("S1", "ADL1"), ("S1", "ADL2"), ("S1", "ADL3"), ("S1", "ADL4"), ("S1", "ADL5"), ("S1", "Drill"),
    ("S2", "ADL1"), ("S2", "ADL2"), ("S2", "ADL3"), ("S2", "Drill"),
    ("S3", "ADL1"), ("S3", "ADL2"), ("S3", "ADL3"), ("S3", "Drill"),
]
DEFAULT_VAL_FILES = [
    ("S2", "ADL4"), ("S2", "ADL5"),
]
DEFAULT_TEST_FILES = [
    ("S3", "ADL4"), ("S3", "ADL5"),
]


def remap_labels_to_continuous_indices(
    y: np.ndarray,
    track: str = "Locomotion",
) -> Tuple[np.ndarray, Dict[int, int], Dict[int, str]]:
    """Remap discrete OPPORTUNITY class labels into continuous 0-indexed integers [0, K-1].

    For Locomotion:
        Original: 0: Null, 1: Stand, 2: Walk, 4: Sit, 5: Lie
        Mapped  : 0: Null, 1: Stand, 2: Walk, 3: Sit, 4: Lie
    """
    unique_classes = sorted(list(np.unique(y)))
    class_map = {int(orig): int(i) for i, orig in enumerate(unique_classes)}
    remapped_y = np.array([class_map[int(val)] for val in y], dtype=np.int64)

    name_map = {}
    if track == "Locomotion":
        raw_names = {0: "Null", 1: "Stand", 2: "Walk", 4: "Sit", 5: "Lie"}
        for orig, mapped in class_map.items():
            name_map[int(mapped)] = raw_names.get(orig, f"Class_{orig}")
    else:
        for orig, mapped in class_map.items():
            name_map[int(mapped)] = f"Gesture_{orig}" if orig != 0 else "Null"

    return remapped_y, class_map, name_map


def preprocess_opportunity_dataset(
    target_track: str = "Locomotion",
    window_length: int = 30,  # 1 second at 30 Hz
    stride: int = 15,         # 50% overlap (0.5 second)
    output_filename: str = "opportunity_preprocessed.npz",
    max_train_files: int = None,  # for quick smoke testing
) -> Path:
    """Preprocess OPPORTUNITY dataset into normalized sliding-window tensors.

    Args:
        target_track (str): 'Locomotion' or 'ML_Both_Arms'.
        window_length (int): Sliding window temporal duration in samples.
        stride (int): Step between consecutive window starting points.
        output_filename (str): Name of output .npz file under data/processed/.
        max_train_files (int): Optional cap on files to speed up test runs.

    Returns:
        Path: Path to saved .npz archive.
    """
    loader = OpportunityLoader(target_track=target_track, sensor_selection="on_body")
    processed_dir = REPO_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_file_list = DEFAULT_TRAIN_FILES[:max_train_files] if max_train_files else DEFAULT_TRAIN_FILES
    val_file_list = DEFAULT_VAL_FILES
    test_file_list = DEFAULT_TEST_FILES

    logger.info("Loading training recordings to compute leak-free normalizer statistics...")
    train_recordings = []
    train_concatenated_data = []

    for sub, r in train_file_list:
        try:
            data, labels, _ = loader.load_recording(sub, r, interpolate_nans=True)
            train_recordings.append((sub, r, data, labels))
            train_concatenated_data.append(data)
        except FileNotFoundError:
            logger.warning("Could not find train recording %s-%s, skipping.", sub, r)

    if not train_concatenated_data:
        raise RuntimeError("No training recordings could be loaded!")

    # Fit scaler STRICTLY on training data observations
    logger.info("Fitting SensorStandardScaler on %d training recordings...", len(train_recordings))
    train_all_stacked = np.concatenate(train_concatenated_data, axis=0)
    scaler = SensorStandardScaler()
    scaler.fit(train_all_stacked)

    # Process recordings into sliding windows
    def process_split_recordings(file_list, is_train=False):
        all_windows = []
        all_labels = []
        all_subjects = []

        recordings_to_load = train_recordings if is_train else None

        if recordings_to_load is not None:
            items = recordings_to_load
        else:
            items = []
            for sub, r in file_list:
                try:
                    data, labels, _ = loader.load_recording(sub, r, interpolate_nans=True)
                    items.append((sub, r, data, labels))
                except FileNotFoundError:
                    logger.warning("Could not find recording %s-%s, skipping.", sub, r)

        for sub, r, raw_sensor_data, raw_labels in items:
            # Apply normalizer
            norm_data = scaler.transform(raw_sensor_data)
            # Create sliding windows for this continuous recording (channel_first=True for (N, C, T))
            windows, w_labels = create_sliding_windows(
                data=norm_data,
                labels=raw_labels,
                window_size=window_length,
                stride=stride,
                label_strategy="mode",
                channel_first=True,
            )
            if len(windows) > 0:
                all_windows.append(windows)
                all_labels.append(w_labels)
                all_subjects.extend([sub] * len(windows))

        if not all_windows:
            empty_shape = (0, 133, window_length)
            return np.empty(empty_shape, dtype=np.float32), np.empty((0,), dtype=np.int64), []

        return np.concatenate(all_windows, axis=0), np.concatenate(all_labels, axis=0), all_subjects

    logger.info("Windowing Training split...")
    X_train, y_train_raw, train_subs = process_split_recordings(train_file_list, is_train=True)

    logger.info("Windowing Validation split...")
    X_val, y_val_raw, val_subs = process_split_recordings(val_file_list, is_train=False)

    logger.info("Windowing Test split...")
    X_test, y_test_raw, test_subs = process_split_recordings(test_file_list, is_train=False)

    # Remap discrete class labels to 0..K-1
    y_train, class_map, name_map = remap_labels_to_continuous_indices(y_train_raw, track=target_track)
    y_val = np.array([class_map.get(val, 0) for val in y_val_raw], dtype=np.int64)
    y_test = np.array([class_map.get(val, 0) for val in y_test_raw], dtype=np.int64)

    output_path = processed_dir / output_filename
    np.savez_compressed(
        output_path,
        X_train=X_train,
        y_train=y_train,
        train_subjects=np.array(train_subs),
        X_val=X_val,
        y_val=y_val,
        val_subjects=np.array(val_subs),
        X_test=X_test,
        y_test=y_test,
        test_subjects=np.array(test_subs),
        class_mapping=json.dumps(class_map),
        class_names=json.dumps(name_map),
        scaler_mean=scaler.mean_,
        scaler_std=scaler.std_,
        track=target_track,
        sampling_rate_hz=30.0,
        window_length=window_length,
        stride=stride,
    )

    logger.info("Successfully saved preprocessed dataset to: %s", output_path)
    logger.info("  Train tensor shape: %s (labels: %s)", X_train.shape, y_train.shape)
    logger.info("  Val tensor shape  : %s (labels: %s)", X_val.shape, y_val.shape)
    logger.info("  Test tensor shape : %s (labels: %s)", X_test.shape, y_test.shape)
    logger.info("  Class mapping: %s", name_map)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Preprocess UCI OPPORTUNITY dataset")
    parser.add_argument("--track", type=str, default="Locomotion", choices=["Locomotion", "ML_Both_Arms"])
    parser.add_argument("--window_length", type=int, default=30, help="Window length in samples (30 = 1 sec at 30 Hz)")
    parser.add_argument("--stride", type=int, default=15, help="Window stride in samples (15 = 50% overlap)")
    parser.add_argument("--output", type=str, default="opportunity_locomotion.npz", help="Output filename")
    parser.add_argument("--max_train_files", type=int, default=None, help="Cap train files for smoke testing")
    args = parser.parse_args()

    preprocess_opportunity_dataset(
        target_track=args.track,
        window_length=args.window_length,
        stride=args.stride,
        output_filename=args.output,
        max_train_files=args.max_train_files,
    )


if __name__ == "__main__":
    main()
