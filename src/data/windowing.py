"""Sliding-window segmentation for continuous multi-channel time-series data.

Provides functions to extract fixed-size temporal windows with configurable
stride/overlap while ensuring windows never cross subject or run boundaries.
"""

from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from scipy import stats


def create_sliding_windows(
    data: np.ndarray,
    labels: np.ndarray,
    window_size: int,
    stride: int,
    label_strategy: str = "mode",
    channel_first: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Segment a continuous multi-channel sequence into fixed sliding windows.

    Args:
        data (np.ndarray): Sensor stream of shape (T_timesteps, C_channels).
        labels (np.ndarray): Corresponding class labels of shape (T_timesteps,).
        window_size (int): Temporal length of each window in samples.
        stride (int): Step size between consecutive window starting points.
        label_strategy (str): How to assign a single class label to a window:
            - 'mode': Most frequent activity label in the window (recommended for HAR).
            - 'last': Label of the last sample in the window.
        channel_first (bool): If True, returns tensor formatted as
            (num_windows, num_channels, window_size) for PyTorch Conv1D.
            If False, returns (num_windows, window_size, num_channels).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - windows: Segmented sensor array of shape (N, C, T) or (N, T, C).
            - window_labels: 1D array of shape (N,) containing integer labels.
    """
    num_samples, num_channels = data.shape
    if num_samples < window_size:
        # Not enough samples for a single window
        empty_shape = (0, num_channels, window_size) if channel_first else (0, window_size, num_channels)
        return np.empty(empty_shape, dtype=data.dtype), np.empty((0,), dtype=labels.dtype)

    start_indices = np.arange(0, num_samples - window_size + 1, stride)
    num_windows = len(start_indices)

    windows = np.zeros((num_windows, window_size, num_channels), dtype=data.dtype)
    window_labels = np.zeros(num_windows, dtype=labels.dtype)

    for i, start in enumerate(start_indices):
        end = start + window_size
        windows[i] = data[start:end, :]

        segment_labels = labels[start:end]
        if label_strategy == "mode":
            mode_result = stats.mode(segment_labels, keepdims=False)
            window_labels[i] = mode_result.mode
        elif label_strategy == "last":
            window_labels[i] = segment_labels[-1]
        else:
            raise ValueError(f"Unknown label strategy: '{label_strategy}'")

    if channel_first:
        # Transpose from (N, T, C) -> (N, C, T)
        windows = np.transpose(windows, (0, 2, 1))

    return windows, window_labels


def window_dataframe_by_group(
    df: pd.DataFrame,
    sensor_columns: List[str],
    label_column: str,
    group_column: str,
    window_size: int,
    stride: int,
    label_strategy: str = "mode",
    channel_first: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Segment DataFrame into sliding windows grouped by continuous segments (e.g. subject or run).

    Prevents windows from spanning across boundaries of different subjects or recording sessions.

    Args:
        df (pd.DataFrame): Time-series dataframe.
        sensor_columns (List[str]): List of sensor channel column names.
        label_column (str): Name of the target activity label column.
        group_column (str): Column indicating continuous recording groups (e.g., 'subject_id' or 'run_id').
        window_size (int): Window duration in samples.
        stride (int): Stride duration in samples.
        label_strategy (str): 'mode' or 'last'.
        channel_first (bool): Output format flag (N, C, T) vs (N, T, C).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Aggregated windows and corresponding labels across all groups.
    """
    all_windows = []
    all_labels = []

    for _, group in df.groupby(group_column, sort=False):
        data = group[sensor_columns].to_numpy(dtype=np.float32)
        labels = group[label_column].to_numpy(dtype=np.int64)

        w, l = create_sliding_windows(
            data=data,
            labels=labels,
            window_size=window_size,
            stride=stride,
            label_strategy=label_strategy,
            channel_first=channel_first,
        )

        if len(w) > 0:
            all_windows.append(w)
            all_labels.append(l)

    if not all_windows:
        empty_shape = (0, len(sensor_columns), window_size) if channel_first else (0, window_size, len(sensor_columns))
        return np.empty(empty_shape, dtype=np.float32), np.empty((0,), dtype=np.int64)

    return np.concatenate(all_windows, axis=0), np.concatenate(all_labels, axis=0)
