"""Data loading, preprocessing, segmentation, splitting, and PyTorch dataset modules.

Note:
    Specific column names, sensor mappings, sampling rates, and labels
    will be populated strictly according to the official OPPORTUNITY
    documentation once the dataset download is complete.
"""

from src.data.synthetic import generate_synthetic_imu_data
from src.data.preprocessing import (
    handle_missing_values,
    SensorStandardScaler,
    SensorMinMaxScaler,
)
from src.data.windowing import create_sliding_windows, window_dataframe_by_group
from src.data.splitting import subject_split, generate_loso_splits
from src.data.loader import HARDataset, create_dataloaders

__all__ = [
    "generate_synthetic_imu_data",
    "handle_missing_values",
    "SensorStandardScaler",
    "SensorMinMaxScaler",
    "create_sliding_windows",
    "window_dataframe_by_group",
    "subject_split",
    "generate_loso_splits",
    "HARDataset",
    "create_dataloaders",
]
