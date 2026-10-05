"""Unit tests for Baseline1DCNN, Trainer engine, evaluation pipeline, and checkpointing.
"""

import sys
from pathlib import Path
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.baseline import Baseline1DCNN
from src.training.trainer import Trainer
from src.training.evaluate import run_inference, evaluate_model


def test_baseline_1d_cnn_initialization_and_forward():
    """Verify Baseline1DCNN initialization and expected forward tensor dimensions (B, C, T) -> (B, K)."""
    batch_size = 12
    in_channels = 16
    sequence_length = 35
    num_classes = 4

    model = Baseline1DCNN(
        in_channels=in_channels,
        num_classes=num_classes,
        sequence_length=sequence_length,
        conv_channels=[16, 32, 64],
        kernel_size=5,
        dropout=0.1,
    )

    x = torch.randn(batch_size, in_channels, sequence_length)
    logits = model(x)

    assert logits.shape == (batch_size, num_classes), f"Expected shape {(batch_size, num_classes)}, got {logits.shape}"
    assert not torch.isnan(logits).any()


def test_baseline_1d_cnn_dimension_validation():
    """Verify Baseline1DCNN raises clear ValueErrors when input dimensions are malformed."""
    model = Baseline1DCNN(in_channels=6, num_classes=3, sequence_length=20)

    # Wrong number of dimensions (2D instead of 3D)
    with pytest.raises(ValueError, match="Expected 3D input tensor"):
        model(torch.randn(8, 20))

    # Wrong number of channels (10 instead of 6)
    with pytest.raises(ValueError, match="Expected 6 input channels"):
        model(torch.randn(8, 10, 20))


def test_trainer_execution_and_checkpoint(tmp_path):
    """Verify Trainer training loop, history tracking, and checkpoint saving/loading."""
    B, C, T, K = 32, 8, 25, 3
    x_train = torch.randn(B, C, T)
    y_train = torch.randint(0, K, (B,))
    x_val = torch.randn(16, C, T)
    y_val = torch.randint(0, K, (16,))

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=8)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=8)

    model = Baseline1DCNN(in_channels=C, num_classes=K, sequence_length=T, conv_channels=[16, 32])
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    ckpt_dir = tmp_path / "checkpoints"
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device="cpu",
        checkpoint_dir=ckpt_dir,
        experiment_name="test_trainer",
        early_stopping_patience=3,
    )

    history = trainer.fit(train_loader, val_loader, epochs=2)

    assert len(history["train_loss"]) == 2
    assert len(history["val_loss"]) == 2
    assert len(history["val_accuracy"]) == 2
    assert len(history["val_macro_f1"]) == 2

    # Verify checkpoint saving
    expected_ckpt = ckpt_dir / "test_trainer_best.pt"
    assert expected_ckpt.exists()

    # Verify checkpoint loading
    new_model = Baseline1DCNN(in_channels=C, num_classes=K, sequence_length=T, conv_channels=[16, 32])
    new_trainer = Trainer(model=new_model, optimizer=optim.Adam(new_model.parameters()), device="cpu")
    payload = new_trainer.load_checkpoint(expected_ckpt)
    assert "model_state_dict" in payload
    assert payload["epoch"] in [1, 2]


def test_evaluate_model_pipeline(tmp_path):
    """Verify evaluate_model runs inference and saves metrics.json and metrics.csv."""
    B, C, T, K = 20, 6, 20, 3
    x_test = torch.randn(B, C, T)
    y_test = torch.randint(0, K, (B,))
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=10)

    model = Baseline1DCNN(in_channels=C, num_classes=K, sequence_length=T, conv_channels=[16, 32])
    out_dir = tmp_path / "eval_out"

    metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        device="cpu",
        output_dir=out_dir,
        experiment_name="test_eval",
        target_names=["A", "B", "C"],
    )

    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "metrics.csv").exists()
    assert (out_dir / "predictions.csv").exists()


def test_cpu_and_optional_cuda_execution():
    """Verify execution on CPU and also CUDA if a GPU is detected."""
    model = Baseline1DCNN(in_channels=4, num_classes=2, sequence_length=15, conv_channels=[8, 16])
    x = torch.randn(4, 4, 15)

    # 1. CPU Test
    model_cpu = model.to("cpu")
    out_cpu = model_cpu(x.to("cpu"))
    assert out_cpu.device.type == "cpu"
    assert out_cpu.shape == (4, 2)

    # 2. CUDA Test (if available)
    if torch.cuda.is_available():
        model_cuda = model.to("cuda")
        out_cuda = model_cuda(x.to("cuda"))
        assert out_cuda.device.type == "cuda"
        assert out_cuda.shape == (4, 2)
