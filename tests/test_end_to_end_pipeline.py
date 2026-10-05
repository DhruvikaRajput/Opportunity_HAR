"""End-to-end integration test for the entire synthetic HAR software pipeline.

Flow:
    synthetic data
         ↓
    loader / tabular split
         ↓
    preprocessing (leak-free normalization)
         ↓
    subject-based split (train / val / test)
         ↓
    sliding windows (N, C, T)
         ↓
    simple baseline classifier
         ↓
    evaluation (accuracy, macro F1, confusion matrix)
         ↓
    saved results & checkpoints
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.seed import seed_everything
from src.utils.device import get_device
from src.data.synthetic import generate_synthetic_imu_data
from src.data.splitting import subject_split
from src.data.preprocessing import SensorStandardScaler
from src.data.windowing import window_dataframe_by_group
from src.data.loader import create_dataloaders
from src.models.simple_baseline import SimpleMLPBaseline
from src.training.engine import train_model, evaluate_epoch


def test_full_pipeline_end_to_end(tmp_path):
    """Verify that all pipeline components execute seamlessly from synthetic raw data to saved results."""
    # 1. Reproducibility
    seed_everything(seed=101, deterministic=True)
    device = get_device("cpu")

    # 2. Synthetic Data
    num_classes = 4
    num_channels = 12
    window_size = 20
    stride = 10

    raw_df, meta = generate_synthetic_imu_data(
        num_subjects=3,
        num_channels=num_channels,
        samples_per_subject=600,
        num_classes=num_classes,
        random_seed=101,
    )
    channels = meta["channel_names"]

    # 3. Subject-based partition
    splits = subject_split(
        raw_df,
        subject_column="subject_id",
        train_subjects=["Subject_01"],
        val_subjects=["Subject_02"],
        test_subjects=["Subject_03"],
    )

    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits["test"]

    # 4. Leak-free Normalization
    scaler = SensorStandardScaler()
    train_df[channels] = scaler.fit_transform(train_df[channels])
    val_df[channels] = scaler.transform(val_df[channels])
    test_df[channels] = scaler.transform(test_df[channels])

    # 5. Sliding-Window Segmentation
    X_train, y_train = window_dataframe_by_group(
        train_df, channels, "activity_label", "subject_id", window_size, stride
    )
    X_val, y_val = window_dataframe_by_group(
        val_df, channels, "activity_label", "subject_id", window_size, stride
    )
    X_test, y_test = window_dataframe_by_group(
        test_df, channels, "activity_label", "subject_id", window_size, stride
    )

    assert X_train.shape[1] == num_channels
    assert X_train.shape[2] == window_size

    # 6. PyTorch DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size=16
    )

    # 7. Baseline Classifier
    model = SimpleMLPBaseline(
        in_channels=num_channels,
        sequence_length=window_size,
        num_classes=num_classes,
        hidden_dim=32,
    )
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # 8. Training loop with early stopping & checkpointing
    checkpoint_dir = tmp_path / "checkpoints"
    best_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=3,
        checkpoint_dir=checkpoint_dir,
        experiment_name="test_pipeline",
    )

    # 9. Evaluation on test partition
    test_loss, test_metrics = evaluate_epoch(best_model, test_loader, criterion, device)

    # Verification of evaluation metrics
    assert "accuracy" in test_metrics
    assert "macro_f1" in test_metrics
    assert "weighted_f1" in test_metrics
    assert "confusion_matrix" in test_metrics
    assert "classification_report" in test_metrics

    # Verify checkpoint creation
    assert (checkpoint_dir / "test_pipeline_best.pt").exists()
