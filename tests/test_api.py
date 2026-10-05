"""Unit tests for src.api module."""

import sys
from pathlib import Path
import pytest
import torch
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.api as api


def test_api_setup_project():
    """Verify setup_project initializes environment and reports correct structure."""
    res = api.setup_project(seed=42)
    assert "repo_root" in res
    assert "device" in res
    assert "cuda_available" in res
    assert res["seed"] == 42


def test_api_build_models():
    """Verify factory functions build models with correct output shapes."""
    cnn = api.build_cnn(in_channels=133, num_classes=5, sequence_length=30)
    fixed_gnn = api.build_fixed_gnn(num_classes=5, sequence_length=30, node_dim=32, gcn_hidden_dim=32)
    adaptive_gnn = api.build_adaptive_gnn(num_classes=5, sequence_length=30, node_dim=32, gcn_hidden_dim=32)

    x = torch.randn(2, 133, 30)
    out_cnn = cnn(x)
    out_fgnn = fixed_gnn(x)
    out_agnn, attn = adaptive_gnn(x, return_attention=True)

    assert out_cnn.shape == (2, 5)
    assert out_fgnn.shape == (2, 5)
    assert out_agnn.shape == (2, 5)
    assert attn.shape == (2, 5, 5)


def test_api_debug_train_mode_cpu_safe():
    """Verify DEBUG mode executes a smoke test safely on CPU."""
    m = api.build_cnn(num_classes=5)
    x = torch.randn(4, 133, 30)
    y = torch.tensor([0, 1, 2, 3])
    ds = torch.utils.data.TensorDataset(x, y)
    dl = torch.utils.data.DataLoader(ds, batch_size=2)

    best_m, hist = api.train_model(m, dl, dl, mode="DEBUG", epochs=1)
    assert best_m is not None
    assert "train_loss" in hist
    assert len(hist["train_loss"]) == 1


def test_api_cuda_guard_stops_full_on_cpu():
    """Verify FULL mode raises RuntimeError when CUDA is not available."""
    if not torch.cuda.is_available():
        m = api.build_cnn()
        dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.randn(2, 133, 30), torch.tensor([0, 1])),
            batch_size=2
        )
        with pytest.raises(RuntimeError, match="CUDA REQUIRED"):
            api.train_model(m, dl, dl, mode="FULL")


def test_api_load_preprocessed_subset():
    """Verify preprocessed subset loading and format."""
    sub = api.load_preprocessed_subset()
    assert "X_train" in sub
    assert "y_train" in sub
    assert "class_names" in sub
    assert isinstance(sub["class_names"], list)
    assert len(sub["class_names"]) == 5


def test_api_locate_dataset():
    """Verify api.locate_dataset returns structured discovery dictionary."""
    res = api.locate_dataset()
    assert "found" in res
    assert "files_found" in res
    assert "total_expected" in res
    assert "dataset_dir" in res
    assert "missing_files" in res
    assert "message" in res
    assert res["total_expected"] == 24
    assert isinstance(res["found"], bool)
    assert isinstance(res["files_found"], int)
    assert isinstance(res["missing_files"], list)


def test_find_opportunity_dataset_missing_dir(tmp_path):
    """Verify dataset discovery reports missing cleanly without errors when files are absent."""
    from src.data.opportunity_loader import find_opportunity_dataset
    import os

    # Force search in an empty directory
    res = find_opportunity_dataset(search_dirs=[tmp_path], repo_root=tmp_path)
    assert res["found"] is False
    assert res["files_found"] == 0
    assert len(res["missing_files"]) == 24
    assert "[MISSING]" in res["message"]


def test_src_data_exports():
    """Verify src.data exposes OpportunityLoader and find_opportunity_dataset."""
    import src.data as sd
    assert hasattr(sd, "OpportunityLoader")
    assert hasattr(sd, "find_opportunity_dataset")
    assert hasattr(sd, "OPPORTUNITY_RECORDING_FILES")
    assert len(sd.OPPORTUNITY_RECORDING_FILES) == 24


