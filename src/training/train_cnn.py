"""Training entry point for 1D CNN baseline on OPPORTUNITY HAR.

COMPUTE SAFETY RULE:
    Checks torch.cuda.is_available().
    If CUDA is unavailable, refuses to launch full CPU training to protect the developer laptop.
    Supports --smoke_test flag to perform a minimal 1-batch verification forward pass on CPU.

Usage:
    # Full GPU training:
    python -m src.training.train_cnn --config configs/cnn.yaml

    # CPU safety check / forward-pass smoke test:
    python -m src.training.train_cnn --config configs/cnn.yaml --smoke_test
"""

import sys
import argparse
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_config
from src.utils.seed import seed_everything
from src.utils.logging import get_logger
from src.utils.device import print_device_info
from src.models.cnn import OpportunityCNN
from src.training.trainer import Trainer
from src.training.evaluate import evaluate_model
from src.data.loader import create_dataloaders
from src.visualization.plots import save_all_experiment_plots

logger = get_logger("train_cnn")


def main():
    parser = argparse.ArgumentParser(description="Train 1D CNN on OPPORTUNITY HAR")
    parser.add_argument("--config", type=str, default="configs/cnn.yaml", help="Path to config file")
    parser.add_argument("--smoke_test", action="store_true", help="Run 1-batch CPU forward-pass verification only")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(seed=cfg.get("seed", 42), deterministic=cfg.get("deterministic", True))

    cuda_available = torch.cuda.is_available()
    dev_name = torch.cuda.get_device_name(0) if cuda_available else "CPU (Laptop)"

    print("=" * 70)
    print(" OPPORTUNITY 1D CNN Baseline Training")
    print("=" * 70)
    print(f"Device:           {'CUDA (GPU)' if cuda_available else 'CPU'}")
    print(f"GPU Name:         {dev_name}")
    print(f"CUDA Available:   {cuda_available}")
    print("=" * 70)

    # STRICT COMPUTE RULE CHECK
    if not cuda_available and not args.smoke_test:
        print("\n[STOPPED - COMPUTE RULE ENFORCED]")
        print("CUDA GPU is not available in this environment.")
        print("As per project instructions, full deep learning training must NOT run on CPU.")
        print("To execute full training:")
        print("  1. Run in Google Colab using notebooks/02_gpu_training.ipynb")
        print("  2. Or run on a GPU-enabled machine with CUDA.")
        print("\nTo verify the pipeline mechanics on CPU without training, pass: --smoke_test\n")
        sys.exit(0)

    # Load data
    data_path = Path(cfg.get("data_path", "data/processed/opportunity_locomotion_subset.npz"))
    if not data_path.is_file():
        raise FileNotFoundError(f"Processed dataset not found at {data_path}. Run preprocess_opportunity first.")

    npz = np.load(data_path, allow_pickle=True)
    X_train = npz["X_train"]
    y_train = npz["y_train"]
    X_val = npz["X_val"]
    y_val = npz["y_val"]
    X_test = npz["X_test"]
    y_test = npz["y_test"]

    class_names_dict = json.loads(str(npz["class_names"]))
    class_names = [class_names_dict.get(str(i), class_names_dict.get(i, f"C{i}")) for i in range(len(class_names_dict))]

    in_channels = cfg.get("in_channels", X_train.shape[1])
    num_classes = cfg.get("num_classes", len(class_names_dict))
    seq_len = cfg.get("sequence_length", X_train.shape[2])

    model = OpportunityCNN(
        in_channels=in_channels,
        num_classes=num_classes,
        sequence_length=seq_len,
        conv_channels=cfg.get("conv_channels", [64, 128, 128]),
        kernel_size=cfg.get("kernel_size", 5),
        dropout=cfg.get("dropout", 0.2),
    )

    if args.smoke_test:
        print("\n[RUNNING CPU SMOKE TEST - 1 MINIBATCH FORWARD PASS]")
        model.eval()
        dummy_input = torch.from_numpy(X_train[:4]).float()
        with torch.no_grad():
            dummy_out = model(dummy_input)
        print(f"  Input shape : {tuple(dummy_input.shape)}")
        print(f"  Output shape: {tuple(dummy_out.shape)} (Expected: (4, {num_classes}))")
        assert dummy_out.shape == (4, num_classes), "Shape mismatch in smoke test!"
        print("  [PASSED] Forward pass successfully verified on CPU without expensive training.\n")
        sys.exit(0)

    # Full GPU Training (executed only when CUDA is active)
    device = torch.device("cuda")
    out_dir = Path(cfg.get("output_dir", "results/deep_learning/cnn"))
    ckpt_dir = Path(cfg.get("checkpoints_dir", "checkpoints"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=cfg.get("batch_size", 64),
        num_workers=cfg.get("num_workers", 2),
        pin_memory=True,
    )

    optimizer = optim.Adam(model.parameters(), lr=cfg.get("learning_rate", 0.001), weight_decay=cfg.get("weight_decay", 0.0001))
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        early_stopping_patience=cfg.get("patience", 8),
        early_stopping_metric="macro_f1",
        checkpoint_dir=ckpt_dir,
        experiment_name="opportunity_cnn",
        clip_grad_norm=cfg.get("clip_grad_norm", 1.0),
    )

    history = trainer.fit(train_loader, val_loader, epochs=cfg.get("epochs", 30))

    best_ckpt = ckpt_dir / "opportunity_cnn_best.pt"
    test_metrics = evaluate_model(
        model=trainer.model,
        test_loader=test_loader,
        checkpoint_path=best_ckpt if best_ckpt.exists() else None,
        device=device,
        output_dir=out_dir,
        experiment_name="opportunity_cnn",
        target_names=class_names,
    )

    save_all_experiment_plots(
        history=history,
        cm=test_metrics["confusion_matrix"],
        output_dir=out_dir,
        class_names=class_names,
    )

    print("Full GPU training complete. Metrics and plots saved to:", out_dir)


if __name__ == "__main__":
    main()
