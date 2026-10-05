"""Generic 1D Convolutional Neural Network baseline for multivariate time-series HAR.

This model is strictly generic and independent of any specific dataset.

EXPECTED TENSOR SHAPES:
    Input  : (batch_size, in_channels, sequence_length)
             batch_size      (B) : Number of time-series window samples in the mini-batch.
             in_channels     (C) : Number of continuous sensor channels / modalities.
             sequence_length (T) : Number of temporal steps in each sliding window.

    Output : (batch_size, num_classes)
             batch_size      (B) : Number of samples matching the input batch.
             num_classes     (K) : Raw unnormalized logit scores for each activity class.
"""

from typing import List, Optional
import torch
import torch.nn as nn


class Baseline1DCNN(nn.Module):
    """Generic 1D CNN baseline for multivariate sensor time-series classification.

    Architecture Overview:
        Input: (B, C, T)
        -> Sequential Conv1D blocks (Conv1D -> BatchNorm1D -> ReLU -> MaxPool1D -> Dropout)
        -> Global Adaptive Average Pooling -> (B, feature_dim, 1)
        -> Flatten -> (B, feature_dim)
        -> Fully Connected Classifier Head -> (B, num_classes)
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        sequence_length: int,
        conv_channels: Optional[List[int]] = None,
        kernel_size: int = 5,
        dropout: float = 0.2,
    ):
        """Initialize Baseline1DCNN.

        Args:
            in_channels (int): Number of input sensor channels (C).
            num_classes (int): Number of target activity classes (K).
            sequence_length (int): Length of input temporal sequence (T).
            conv_channels (Optional[List[int]]): Filters per conv layer. Defaults to [32, 64, 128].
            kernel_size (int): 1D temporal filter kernel size. Defaults to 5.
            dropout (float): Dropout probability for regularization. Defaults to 0.2.
        """
        super().__init__()
        if conv_channels is None:
            conv_channels = [32, 64, 128]

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.conv_channels = conv_channels

        layers = []
        current_channels = in_channels
        padding = kernel_size // 2

        for i, out_c in enumerate(conv_channels):
            layers.append(
                nn.Conv1d(
                    in_channels=current_channels,
                    out_channels=out_c,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm1d(out_c))
            layers.append(nn.ReLU(inplace=True))

            # Apply temporal pooling on intermediate layers
            if i < len(conv_channels) - 1:
                layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
                layers.append(nn.Dropout(p=dropout))

            current_channels = out_c

        # Global average pooling reduces arbitrary temporal length to 1
        layers.append(nn.AdaptiveAvgPool1d(output_size=1))
        self.feature_extractor = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(conv_channels[-1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, time).

        Returns:
            torch.Tensor: Unnormalized class logits of shape (batch, num_classes).
        """
        if x.dim() != 3:
            raise ValueError(
                f"Expected 3D input tensor of shape (batch, channels, time), but received shape {tuple(x.shape)}"
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, but received {x.shape[1]} (shape: {tuple(x.shape)})"
            )

        # (B, C, T) -> (B, conv_channels[-1], 1)
        feat = self.feature_extractor(x)
        # (B, conv_channels[-1], 1) -> (B, conv_channels[-1])
        feat_flat = feat.squeeze(-1)
        # (B, conv_channels[-1]) -> (B, num_classes)
        logits = self.classifier(feat_flat)
        return logits
