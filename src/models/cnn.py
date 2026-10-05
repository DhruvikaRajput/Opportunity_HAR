"""1D Convolutional Neural Network architecture for OPPORTUNITY HAR.

TENSOR SHAPES:
    Input  : (batch_size, in_channels, sequence_length)
             batch_size      (B) : Number of sliding windows in the mini-batch
             in_channels     (C) : 133 on-body sensor channels
             sequence_length (T) : Temporal window length (e.g. 30 samples = 1 sec at 30 Hz)
    Output : (batch_size, num_classes)
             num_classes     (K) : 5 classes for Locomotion (Null, Stand, Walk, Sit, Lie)
"""

from typing import List, Optional
import torch
import torch.nn as nn


class OpportunityCNN(nn.Module):
    """1D CNN baseline for OPPORTUNITY multi-sensor temporal windows.

    Architecture:
        Input: (B, 133, T)
        -> Conv1D(133, 64, kernel=5) -> BatchNorm1D -> ReLU -> MaxPool1D(2) -> Dropout(0.2)
        -> Conv1D(64, 128, kernel=5) -> BatchNorm1D -> ReLU -> MaxPool1D(2) -> Dropout(0.2)
        -> Conv1D(128, 128, kernel=3) -> BatchNorm1D -> ReLU -> AdaptiveAvgPool1D(1)
        -> Flatten -> Linear(128, num_classes)
    """

    def __init__(
        self,
        in_channels: int = 133,
        num_classes: int = 5,
        sequence_length: int = 30,
        conv_channels: Optional[List[int]] = None,
        kernel_size: int = 5,
        dropout: float = 0.2,
    ):
        super().__init__()
        if conv_channels is None:
            conv_channels = [64, 128, 128]

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.sequence_length = sequence_length

        layers = []
        cur_in = in_channels
        padding = kernel_size // 2

        for i, out_c in enumerate(conv_channels):
            k = kernel_size if i < len(conv_channels) - 1 else 3
            p = k // 2
            layers.append(
                nn.Conv1d(cur_in, out_c, kernel_size=k, padding=p, bias=False)
            )
            layers.append(nn.BatchNorm1d(out_c))
            layers.append(nn.ReLU(inplace=True))

            if i < len(conv_channels) - 1:
                layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
                layers.append(nn.Dropout(p=dropout))

            cur_in = out_c

        layers.append(nn.AdaptiveAvgPool1d(1))
        self.feature_extractor = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(conv_channels[-1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): (batch_size, in_channels, sequence_length).

        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes).
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D tensor (B, C, T), got {x.shape}")
        if x.size(1) != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {x.size(1)}")

        feat = self.feature_extractor(x)  # (B, 128, 1)
        feat_flat = feat.squeeze(-1)       # (B, 128)
        logits = self.classifier(feat_flat)  # (B, num_classes)
        return logits
