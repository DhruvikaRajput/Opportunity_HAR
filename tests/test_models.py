"""Unit tests for deep learning models: BaselineCNN forward pass, tensor shapes, and gradients.
"""

import sys
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.baseline_cnn import BaselineCNN


def test_baseline_cnn_forward_shapes():
    """Verify BaselineCNN produces exact expected output shape (batch_size, num_classes)."""
    batch_size = 16
    in_channels = 18
    sequence_length = 30
    num_classes = 5

    model = BaselineCNN(
        in_channels=in_channels,
        num_classes=num_classes,
        conv_filters=[32, 64, 64],
        kernel_size=5,
        dropout=0.1,
    )

    x = torch.randn(batch_size, in_channels, sequence_length)
    logits = model(x)

    assert logits.shape == (batch_size, num_classes)
    assert not torch.isnan(logits).any(), "NaN detected in model output logits!"


def test_baseline_cnn_gradient_flow():
    """Verify backpropagation produces valid gradients through all layers."""
    batch_size = 8
    in_channels = 6
    sequence_length = 20
    num_classes = 3

    model = BaselineCNN(
        in_channels=in_channels,
        num_classes=num_classes,
        conv_filters=[16, 32],
        kernel_size=3,
    )

    x = torch.randn(batch_size, in_channels, sequence_length)
    y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])

    criterion = torch.nn.CrossEntropyLoss()
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient!"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradient!"
