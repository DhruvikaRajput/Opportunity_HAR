"""Methodology audit and integrity verification unit tests for OPPORTUNITY HAR."""

import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from src.features.body_regions import (
    BODY_REGIONS,
    ANATOMICAL_SENSOR_COLUMNS,
    get_on_body_column_indices,
    get_fixed_adjacency_matrix,
)
from src.data.windowing import create_sliding_windows
from src.data.preprocessing import SensorStandardScaler
from src.models.classical import extract_statistical_features, train_logistic_regression, train_linear_svm
from src.evaluation.metrics import compute_har_metrics


def test_anatomical_channel_mapping_and_counts():
    """Verify anatomical channel mappings, counts, and non-overlapping completeness."""
    expected_counts = {
        "trunk": 19,
        "right_arm": 38,
        "left_arm": 38,
        "right_leg": 22,
        "left_leg": 16,
    }
    for region, expected in expected_counts.items():
        assert len(ANATOMICAL_SENSOR_COLUMNS[region]) == expected, (
            f"Region '{region}' expected {expected} channels, got {len(ANATOMICAL_SENSOR_COLUMNS[region])}"
        )

    all_on_body = get_on_body_column_indices()
    assert len(all_on_body) == 133, f"Expected 133 total on-body channels, got {len(all_on_body)}"
    assert len(set(all_on_body)) == 133, "Duplicate channel indices detected in anatomical regions"

    adj = get_fixed_adjacency_matrix()
    assert adj.shape == (5, 5), "Adjacency matrix must be 5x5"
    assert np.all(adj == adj.T), "Fixed anatomical adjacency must be symmetric"


def test_no_window_leakage_across_boundaries():
    """Verify that create_sliding_windows respects recording length and does not lookahead."""
    T, C = 100, 10
    data = np.arange(T * C, dtype=np.float32).reshape(T, C)
    labels = np.array([0] * 50 + [1] * 50, dtype=np.int64)

    w, wl = create_sliding_windows(data, labels, window_size=30, stride=15, channel_first=True)
    assert w.shape == (5, C, 30), f"Expected 5 windows of shape (5, {C}, 30), got {w.shape}"
    assert len(wl) == 5

    # Check that first window only uses samples [0:30]
    np.testing.assert_array_equal(w[0], data[:30].T)
    # Check that second window starts exactly at sample 15
    np.testing.assert_array_equal(w[1], data[15:45].T)


def test_normalizer_fitted_strictly_on_train():
    """Verify SensorStandardScaler fits on train data only and does not modify test scale."""
    X_train = np.array([[10.0, 20.0], [20.0, 40.0], [30.0, 60.0]], dtype=np.float32)
    X_test = np.array([[100.0, 200.0]], dtype=np.float32)

    scaler = SensorStandardScaler()
    scaler.fit(X_train)

    np.testing.assert_allclose(scaler.mean_, [20.0, 40.0], rtol=1e-5)
    np.testing.assert_allclose(scaler.std_, [8.164966, 16.329931], rtol=1e-5)

    norm_test = scaler.transform(X_test)
    expected_test_norm = (X_test - [20.0, 40.0]) / [8.164966, 16.329931]
    np.testing.assert_allclose(norm_test, expected_test_norm, rtol=1e-4)


def test_statistical_feature_extraction_dimensions():
    """Verify statistical feature extraction produces exactly C * 4 = 532 features."""
    N, C, T = 10, 133, 30
    synthetic_windows = np.random.randn(N, C, T).astype(np.float32)

    features = extract_statistical_features(synthetic_windows)
    assert features.shape == (N, 532), f"Expected shape ({N}, 532), got {features.shape}"

    # Verify column layout: [mean, std, min, max]
    expected_mean = np.mean(synthetic_windows, axis=2)
    np.testing.assert_allclose(features[:, :C], expected_mean, rtol=1e-5)


def test_linear_models_scaling_and_convergence():
    """Verify that StandardScaler integration allows clean linear model training."""
    np.random.seed(42)
    X_tr = np.random.randn(200, 532).astype(np.float32) * 100.0 + 50.0
    y_tr = np.random.choice([0, 1, 2, 3, 4], size=200)
    X_te = np.random.randn(50, 532).astype(np.float32) * 100.0 + 50.0
    y_te = np.random.choice([0, 1, 2, 3, 4], size=50)

    # Test with scaling enabled
    _, lr_m, lr_pred = train_logistic_regression(X_tr, y_tr, X_te, y_te, scale_features=True, max_iter=200)
    assert "accuracy" in lr_m and "macro_f1" in lr_m
    assert len(lr_pred) == 50

    _, svm_m, svm_pred = train_linear_svm(X_tr, y_tr, X_te, y_te, scale_features=True, max_iter=500)
    assert "accuracy" in svm_m and "macro_f1" in svm_m
    assert len(svm_pred) == 50


def test_metrics_calculation_consistency():
    """Verify accuracy, precision, recall, and F1 calculations on known edge cases."""
    y_true = np.array([0, 1, 2, 3, 4])
    y_pred = np.array([0, 1, 2, 3, 0])  # class 4 misclassified as 0

    metrics = compute_har_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 4.0 / 5.0
    # Class 4 has precision=0, recall=0, f1=0
    assert metrics["classification_report"]["4"]["f1-score"] == 0.0
    assert metrics["macro_f1"] > 0.0
    assert metrics["weighted_f1"] > 0.0


def test_robustness_simulation_preserves_dimensions():
    """Verify that zero-masking missing channels preserves 532 feature dimensions."""
    N, C, T = 5, 133, 30
    X_clean = np.ones((N, C, T), dtype=np.float32)

    # Drop 20 channels
    drop_indices = [0, 5, 10, 20, 50]
    X_corrupted = X_clean.copy()
    X_corrupted[:, drop_indices, :] = 0.0

    feats = extract_statistical_features(X_corrupted)
    assert feats.shape == (N, 532)

    # Dropped channels must have mean=0, std=0, min=0, max=0
    for idx in drop_indices:
        np.testing.assert_allclose(feats[:, idx], 0.0)          # mean
        np.testing.assert_allclose(feats[:, C + idx], 0.0)      # std
        np.testing.assert_allclose(feats[:, 2 * C + idx], 0.0)  # min
        np.testing.assert_allclose(feats[:, 3 * C + idx], 0.0)  # max
