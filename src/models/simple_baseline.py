"""Simple baseline classifiers for software verification and baseline benchmarking.

TENSOR SHAPES:
    Input  : (batch_size, in_channels, sequence_length) or (batch_size, features)
    Output : (batch_size, num_classes)
"""

from typing import Union
import torch
import torch.nn as nn


class SimpleLinearBaseline(nn.Module):
    """Linear baseline classifier that flattens time-series windows directly into a single layer.

    Used as the most minimal deep learning baseline and pipeline sanity check.
    """

    def __init__(self, in_channels: int, sequence_length: int, num_classes: int):
        super().__init__()
        self.in_features = in_channels * sequence_length
        self.classifier = nn.Linear(self.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Tensor of shape (B, C, T) or (B, C * T).

        Returns:
            torch.Tensor: Logits of shape (B, num_classes).
        """
        if x.dim() == 3:
            # Flatten (B, C, T) -> (B, C * T)
            x = x.flatten(start_dim=1)
        return self.classifier(x)


class SimpleMLPBaseline(nn.Module):
    """Multi-Layer Perceptron baseline with one hidden layer and dropout.

    Architecture:
        Input: (B, C, T) -> Flatten (B, C * T)
        -> Linear(C * T, hidden_dim) -> ReLU -> Dropout
        -> Linear(hidden_dim, num_classes)
    """

    def __init__(
        self,
        in_channels: int,
        sequence_length: int,
        num_classes: int,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.in_features = in_channels * sequence_length
        self.net = nn.Sequential(
            nn.Linear(self.in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Tensor of shape (B, C, T) or (B, C * T).

        Returns:
            torch.Tensor: Logits of shape (B, num_classes).
        """
        if x.dim() == 3:
            x = x.flatten(start_dim=1)
        return self.net(x)
