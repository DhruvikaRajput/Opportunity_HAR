"""Baseline 1D Convolutional Neural Network for multi-channel sensor-based HAR.

TENSOR SHAPES:
    Input  : (batch_size, in_channels, sequence_length)
             batch_size (B)       : Number of sliding windows in the mini-batch
             in_channels (C)      : Number of sensor channels (e.g. 18 channels)
             sequence_length (T)  : Temporal samples per window (e.g. 30 samples)
    Output : (batch_size, num_classes)
             num_classes (K)      : Unnormalized logit scores for activity classes
"""

from typing import List
import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    """Configurable 1D CNN for temporal sensor feature extraction and activity classification.

    Architecture:
        Input: (B, C, T)
        -> Block 1: Conv1D(C, F1, kernel) -> BatchNorm1D -> ReLU -> MaxPool1D -> Dropout
        -> Block 2: Conv1D(F1, F2, kernel) -> BatchNorm1D -> ReLU -> MaxPool1D -> Dropout
        -> Block 3: Conv1D(F2, F3, kernel) -> BatchNorm1D -> ReLU -> AdaptiveAvgPool1D(1)
        -> Flatten: (B, F3)
        -> Linear Classifier Head: Linear(F3, num_classes)
        Output: (B, num_classes)
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        conv_filters: List[int] = None,
        kernel_size: int = 5,
        dropout: float = 0.3,
    ):
        """Initialize BaselineCNN.

        Args:
            in_channels (int): Number of input sensor channels (C).
            num_classes (int): Number of target activity classes (K).
            conv_filters (List[int]): Number of filters in sequential conv layers.
                Defaults to [64, 128, 128].
            kernel_size (int): 1D convolution kernel temporal size. Defaults to 5.
            dropout (float): Dropout probability. Defaults to 0.3.
        """
        super().__init__()
        if conv_filters is None:
            conv_filters = [64, 128, 128]

        self.in_channels = in_channels
        self.num_classes = num_classes

        layers = []
        current_in = in_channels
        padding = kernel_size // 2

        for i, out_channels in enumerate(conv_filters):
            layers.append(
                nn.Conv1d(
                    in_channels=current_in,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU(inplace=True))

            # Apply pooling on earlier blocks
            if i < len(conv_filters) - 1:
                layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
                layers.append(nn.Dropout(p=dropout))

            current_in = out_channels

        # Global average pool down to (B, out_channels, 1)
        layers.append(nn.AdaptiveAvgPool1d(output_size=1))
        self.feature_extractor = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(conv_filters[-1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, sequence_length).

        Returns:
            torch.Tensor: Unnormalized class logits of shape (batch_size, num_classes).
        """
        # Feature extraction: (B, C, T) -> (B, F_last, 1)
        feat = self.feature_extractor(x)
        # Flatten: (B, F_last, 1) -> (B, F_last)
        feat_flat = feat.squeeze(-1)
        # Classification: (B, F_last) -> (B, num_classes)
        logits = self.classifier(feat_flat)
        return logits
