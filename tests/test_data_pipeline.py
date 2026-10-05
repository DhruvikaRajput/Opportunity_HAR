"""Unit tests for the generic data pipeline: synthetic generation, preprocessing,
windowing, subject splitting, and PyTorch DataLoaders.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.synthetic import generate_synthetic_imu_data
from src.data.preprocessing import (
    handle_missing_values,
    SensorStandardScaler,
    SensorMinMaxScaler,
)
from src.data.windowing import create_sliding_windows, window_dataframe_by_group
from src.data.splitting import subject_split, generate_loso_splits
from src.data.loader import HARDataset, create_dataloaders


def test_synthetic_imu_generation():
    """Verify synthetic IMU data generation matches specified parameters."""
    num_subs = 3
    num_chans = 12
    samples_per_sub = 500
    num_classes = 4

    df, meta = generate_synthetic_imu_data(
        num_subjects=num_subs,
        num_channels=num_chans,
        samples_per_subject=samples_per_sub,
        num_classes=num_classes,
        random_seed=42,
    )

    assert len(df) == num_subs * samples_per_sub
    assert meta["num_subjects"] == num_subs
    assert meta["num_channels"] == num_chans
    assert len(meta["channel_names"]) == num_chans
    assert df["subject_id"].nunique() == num_subs
    assert set(df["activity_label"].unique()).issubset(set(range(num_classes)))


def test_missing_value_imputation():
    """Verify linear and forward-fill imputation handles missing values."""
    df = pd.DataFrame({
        "sensor_0": [1.0, np.nan, 3.0, np.nan, 5.0],
        "sensor_1": [np.nan, 2.0, np.nan, 4.0, 5.0],
    })
    cleaned = handle_missing_values(df, sensor_columns=["sensor_0", "sensor_1"], strategy="linear")
    assert not cleaned.isna().any().any()
    assert np.isclose(cleaned.loc[1, "sensor_0"], 2.0)


def test_scaler_no_leakage():
    """Verify that SensorStandardScaler statistics are fitted only on train observations."""
    train_data = np.array([[10.0, 100.0], [20.0, 200.0]], dtype=np.float32)
    test_data = np.array([[30.0, 300.0]], dtype=np.float32)

    scaler = SensorStandardScaler()
    scaled_train = scaler.fit_transform(train_data)

    # Train mean should be [15, 150]
    np.testing.assert_allclose(scaler.mean_, [15.0, 150.0])
    # Scaled train mean should be ~0
    np.testing.assert_allclose(scaled_train.mean(axis=0), [0.0, 0.0], atol=1e-5)

    # Transform test without updating mean/std
    scaled_test = scaler.transform(test_data)
    assert scaler.mean_[0] == 15.0, "Scaler statistics leaked during test transform!"
    assert scaled_test.shape == test_data.shape


def test_sliding_window_shapes_and_channel_first():
    """Verify sliding-window segmentation shapes for PyTorch Conv1D (N, C, T)."""
    T = 100
    C = 6
    window_size = 20
    stride = 10

    data = np.random.randn(T, C).astype(np.float32)
    labels = np.random.randint(0, 3, size=T)

    windows, w_labels = create_sliding_windows(
        data=data,
        labels=labels,
        window_size=window_size,
        stride=stride,
        label_strategy="mode",
        channel_first=True,
    )

    expected_windows = (T - window_size) // stride + 1
    assert windows.shape == (expected_windows, C, window_size)
    assert len(w_labels) == expected_windows


def test_subject_split_leak_prevention():
    """Verify that subject splitting strictly isolates subjects with zero overlap."""
    df, _ = generate_synthetic_imu_data(num_subjects=4, samples_per_subject=100)

    splits = subject_split(
        df,
        subject_column="subject_id",
        train_subjects=["Subject_01", "Subject_02"],
        val_subjects=["Subject_03"],
        test_subjects=["Subject_04"],
    )

    train_subs = set(splits["train"]["subject_id"].unique())
    val_subs = set(splits["val"]["subject_id"].unique())
    test_subs = set(splits["test"]["subject_id"].unique())

    assert train_subs.isdisjoint(val_subs)
    assert train_subs.isdisjoint(test_subs)
    assert val_subs.isdisjoint(test_subs)

    # Overlap error detection
    with pytest.raises(ValueError, match="Subject leakage detected"):
        subject_split(
            df,
            subject_column="subject_id",
            train_subjects=["Subject_01"],
            test_subjects=["Subject_01"],
        )


def test_loso_split_generation():
    """Verify Leave-One-Subject-Out generator iterates over all subjects."""
    subs = ["S1", "S2", "S3"]
    splits = list(generate_loso_splits(subs))
    assert len(splits) == 3
    for s in splits:
        assert s["test_subject"] not in s["train_subjects"]
        assert len(s["train_subjects"]) == 2


def test_dataloaders_batching():
    """Verify PyTorch DataLoader yields correctly shaped batches."""
    N = 50
    C = 8
    T = 25
    windows = np.random.randn(N, C, T).astype(np.float32)
    labels = np.random.randint(0, 4, size=N)

    train_loader, _, _ = create_dataloaders(
        train_windows=windows,
        train_labels=labels,
        batch_size=16,
    )

    batch_x, batch_y = next(iter(train_loader))
    assert batch_x.shape == (16, C, T)
    assert batch_y.shape == (16,)
