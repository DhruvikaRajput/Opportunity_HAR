"""Data cleaning, missing value imputation, and strictly leak-free sensor normalizers.

RESEARCH RULE:
    Normalization and scaling statistics must be fitted on TRAINING data ONLY,
    and subsequently applied (transformed) onto validation and test data.
"""

from typing import List, Optional, Union
import numpy as np
import pandas as pd


def handle_missing_values(
    df: pd.DataFrame,
    sensor_columns: List[str],
    strategy: str = "linear",
) -> pd.DataFrame:
    """Impute or interpolate missing sensor readings in continuous time series.

    Args:
        df (pd.DataFrame): Time-series DataFrame.
        sensor_columns (List[str]): Columns containing sensor channels.
        strategy (str): Imputation strategy ('linear', 'forward_fill', 'zero').

    Returns:
        pd.DataFrame: Cleaned DataFrame with imputed sensor values.
    """
    df_clean = df.copy()

    if strategy == "linear":
        df_clean[sensor_columns] = df_clean[sensor_columns].interpolate(
            method="linear", limit_direction="both"
        )
        # Any remaining NaNs at boundaries fill with 0
        df_clean[sensor_columns] = df_clean[sensor_columns].fillna(0.0)
    elif strategy == "forward_fill":
        df_clean[sensor_columns] = df_clean[sensor_columns].ffill().bfill().fillna(0.0)
    elif strategy == "zero":
        df_clean[sensor_columns] = df_clean[sensor_columns].fillna(0.0)
    else:
        raise ValueError(f"Unknown missing value strategy: '{strategy}'")

    return df_clean


class SensorStandardScaler:
    """Z-score normalizer fitted exclusively on training sensor channels to prevent leakage.

    Formula:
        z = (x - mean) / (std + eps)
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> "SensorStandardScaler":
        """Compute mean and standard deviation from training observations only.

        Args:
            X (Union[np.ndarray, pd.DataFrame]): Training sensor data (N_samples, C_channels).

        Returns:
            SensorStandardScaler: Fitted scaler instance.
        """
        arr = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        self.mean_ = np.nanmean(arr, axis=0)
        self.std_ = np.nanstd(arr, axis=0)
        # Replace zero standard deviation with 1.0 to avoid division by zero
        self.std_[self.std_ == 0.0] = 1.0
        self.is_fitted = True
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Apply the previously fitted mean and std to sensor data.

        Args:
            X (Union[np.ndarray, pd.DataFrame]): Sensor data to normalize.

        Returns:
            np.ndarray: Normalized array of same shape.

        Raises:
            RuntimeError: If transform is called prior to fit.
        """
        if not self.is_fitted or self.mean_ is None or self.std_ is None:
            raise RuntimeError(
                "SensorStandardScaler must be fitted on training data before transforming!"
            )
        arr = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        return (arr - self.mean_) / (self.std_ + self.eps)

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Convenience method to fit and transform on training data."""
        return self.fit(X).transform(X)


class SensorMinMaxScaler:
    """Min-Max normalizer fitted exclusively on training sensor channels to scale into [0, 1].

    Formula:
        x_scaled = (x - min) / (max - min + eps)
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.min_: Optional[np.ndarray] = None
        self.max_: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> "SensorMinMaxScaler":
        """Compute min and max from training observations only."""
        arr = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        self.min_ = np.nanmin(arr, axis=0)
        self.max_ = np.nanmax(arr, axis=0)
        self.is_fitted = True
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Apply min-max scaling to sensor data."""
        if not self.is_fitted or self.min_ is None or self.max_ is None:
            raise RuntimeError("SensorMinMaxScaler must be fitted before transforming!")
        arr = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        range_ = self.max_ - self.min_
        range_[range_ == 0.0] = 1.0
        return (arr - self.min_) / (range_ + self.eps)

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Fit and transform training observations."""
        return self.fit(X).transform(X)
