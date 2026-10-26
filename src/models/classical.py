"""Classical machine learning baseline models for HAR (CPU-based).

Extracts per-channel summary statistics (mean, std, min, max) across temporal windows
and trains standard classifiers:
    1. Random Forest
    2. Support Vector Machine (Linear / RBF)
    3. Logistic Regression
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from src.evaluation.metrics import compute_har_metrics


def extract_statistical_features(X_windows: np.ndarray) -> np.ndarray:
    """Extract standard temporal statistics per sensor channel for classical ML.

    Args:
        X_windows (np.ndarray): Tensor of shape (N_samples, C_channels, T_timesteps).

    Returns:
        np.ndarray: Feature matrix of shape (N_samples, C_channels * 4):
            [mean, std, min, max] per channel.
    """
    if X_windows.ndim != 3:
        raise ValueError(f"Expected 3D tensor (N, C, T), got {X_windows.shape}")

    mean_f = np.mean(X_windows, axis=2)
    std_f = np.std(X_windows, axis=2)
    min_f = np.min(X_windows, axis=2)
    max_f = np.max(X_windows, axis=2)

    features = np.hstack([mean_f, std_f, min_f, max_f])
    return features.astype(np.float32)


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_estimators: int = 100,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Tuple[RandomForestClassifier, Dict[str, Any], np.ndarray]:
    """Train and evaluate a Random Forest baseline.

    Returns:
        Tuple: (fitted_model, metrics_dict, y_pred)
    """
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
        max_depth=16,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    metrics = compute_har_metrics(y_test, y_pred)
    return clf, metrics, y_pred


from sklearn.preprocessing import StandardScaler


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42,
    max_iter: int = 500,
    scale_features: bool = False,
) -> Tuple[Any, Dict[str, Any], np.ndarray]:
    """Train and evaluate a Logistic Regression baseline.

    Args:
        X_train, y_train, X_test, y_test: Feature matrices and ground-truth labels.
        random_state (int): Seed for reproducibility.
        max_iter (int): Maximum optimization iterations.
        scale_features (bool): If True, standardizes features with StandardScaler
            fitted exclusively on X_train. Defaults to False for exact replication
            of the raw statistical baseline.

    Returns:
        Tuple: (fitted_model, metrics_dict, y_pred)
    """
    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_te = scaler.transform(X_test)
    else:
        X_tr = X_train
        X_te = X_test

    clf = LogisticRegression(
        max_iter=max_iter,
        random_state=random_state,
        solver="lbfgs",
    )
    clf.fit(X_tr, y_train)
    y_pred = clf.predict(X_te)
    metrics = compute_har_metrics(y_test, y_pred)
    model = (scaler, clf) if scale_features else clf
    return model, metrics, y_pred


def train_linear_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42,
    max_iter: int = 1000,
    scale_features: bool = False,
) -> Tuple[Any, Dict[str, Any], np.ndarray]:
    """Train and evaluate a Linear Support Vector Machine baseline.

    Args:
        X_train, y_train, X_test, y_test: Feature matrices and ground-truth labels.
        random_state (int): Seed for reproducibility.
        max_iter (int): Maximum optimization iterations.
        scale_features (bool): If True, standardizes features with StandardScaler
            fitted exclusively on X_train. Defaults to False for exact replication
            of the raw statistical baseline.

    Returns:
        Tuple: (fitted_model, metrics_dict, y_pred)
    """
    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_te = scaler.transform(X_test)
    else:
        X_tr = X_train
        X_te = X_test

    clf = LinearSVC(
        random_state=random_state,
        max_iter=max_iter,
        dual="auto",
    )
    clf.fit(X_tr, y_train)
    y_pred = clf.predict(X_te)
    metrics = compute_har_metrics(y_test, y_pred)
    model = (scaler, clf) if scale_features else clf
    return model, metrics, y_pred
