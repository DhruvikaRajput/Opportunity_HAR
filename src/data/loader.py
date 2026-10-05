"""PyTorch Dataset and DataLoader wrappers for HAR windowed sensor data.
"""

from typing import Tuple, Optional, Union
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class HARDataset(Dataset):
    """PyTorch Dataset for windowed multi-channel sensor time-series.

    Attributes:
        windows (torch.Tensor): Windowed sensor signals of shape
            (N_samples, C_channels, T_timesteps) or (N_samples, T_timesteps, C_channels).
        labels (torch.Tensor): Activity labels of shape (N_samples,).
    """

    def __init__(
        self,
        windows: Union[np.ndarray, torch.Tensor],
        labels: Union[np.ndarray, torch.Tensor],
    ):
        if isinstance(windows, np.ndarray):
            self.windows = torch.from_numpy(windows).float()
        else:
            self.windows = windows.float()

        if isinstance(labels, np.ndarray):
            self.labels = torch.from_numpy(labels).long()
        else:
            self.labels = labels.long()

        assert len(self.windows) == len(self.labels), (
            f"Mismatched count: {len(self.windows)} windows vs {len(self.labels)} labels"
        )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.windows[idx], self.labels[idx]


def create_dataloaders(
    train_windows: np.ndarray,
    train_labels: np.ndarray,
    val_windows: Optional[np.ndarray] = None,
    val_labels: Optional[np.ndarray] = None,
    test_windows: Optional[np.ndarray] = None,
    test_labels: Optional[np.ndarray] = None,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """Build PyTorch DataLoaders for train, validation, and test splits.

    Args:
        train_windows (np.ndarray): Training windows (N_train, C, T).
        train_labels (np.ndarray): Training labels (N_train,).
        val_windows (Optional[np.ndarray]): Validation windows (N_val, C, T).
        val_labels (Optional[np.ndarray]): Validation labels (N_val,).
        test_windows (Optional[np.ndarray]): Test windows (N_test, C, T).
        test_labels (Optional[np.ndarray]): Test labels (N_test,).
        batch_size (int): Mini-batch size.
        num_workers (int): DataLoader worker count.
        pin_memory (bool): Pin memory for faster CUDA transfer.

    Returns:
        Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
            (train_loader, val_loader, test_loader).
    """
    train_dataset = HARDataset(train_windows, train_labels)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    val_loader = None
    if val_windows is not None and val_labels is not None and len(val_windows) > 0:
        val_dataset = HARDataset(val_windows, val_labels)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    test_loader = None
    if test_windows is not None and test_labels is not None and len(test_windows) > 0:
        test_dataset = HARDataset(test_windows, test_labels)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    return train_loader, val_loader, test_loader
