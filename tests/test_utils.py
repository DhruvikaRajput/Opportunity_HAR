"""Unit tests for utility modules: seed, device, logging, and configuration.
"""

import sys
from pathlib import Path
import random
import numpy as np
import torch

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.seed import seed_everything
from src.utils.device import get_device, print_device_info
from src.utils.logging import get_logger
from src.utils.config import load_config, save_config


def test_seed_everything_reproducibility():
    """Verify that seed_everything yields identical pseudorandom draws across libraries."""
    seed = 1234
    
    # Run 1
    seed_everything(seed=seed, deterministic=True)
    py_val_1 = random.random()
    np_val_1 = np.random.rand(5)
    torch_val_1 = torch.rand(5)

    # Run 2
    seed_everything(seed=seed, deterministic=True)
    py_val_2 = random.random()
    np_val_2 = np.random.rand(5)
    torch_val_2 = torch.rand(5)

    assert py_val_1 == py_val_2, "Python random seed mismatch"
    assert np.allclose(np_val_1, np_val_2), "NumPy random seed mismatch"
    assert torch.allclose(torch_val_1, torch_val_2), "PyTorch random seed mismatch"


def test_device_selection_cpu():
    """Verify explicit CPU device selection."""
    device = get_device(preferred_device="cpu")
    assert device.type == "cpu"
    info = print_device_info(device)
    assert info["device_type"] == "cpu"


def test_device_selection_auto():
    """Verify auto device selection returns a valid PyTorch device (CPU or CUDA)."""
    device = get_device(preferred_device="auto")
    assert device.type in ["cpu", "cuda"]
    if torch.cuda.is_available():
        assert device.type == "cuda"
    else:
        assert device.type == "cpu"


def test_logging_creation(tmp_path):
    """Verify logger creation and log file generation."""
    test_log_dir = tmp_path / "logs"
    logger = get_logger(name="test_logger", log_dir=test_log_dir, log_filename="test.log")
    test_msg = "Research pipeline test log entry."
    logger.info(test_msg)

    log_file = test_log_dir / "test.log"
    assert log_file.exists(), "Log file was not created"
    content = log_file.read_text(encoding="utf-8")
    assert test_msg in content, "Log message missing from log file"


def test_config_loading_and_saving(tmp_path):
    """Verify default YAML config can be loaded and resaved accurately."""
    default_config_path = REPO_ROOT / "configs" / "default_config.yaml"
    assert default_config_path.is_file(), "default_config.yaml does not exist"

    cfg = load_config(default_config_path)
    assert "reproducibility" in cfg
    assert "compute" in cfg
    assert cfg["reproducibility"]["seed"] == 42

    # Test saving
    temp_cfg_path = tmp_path / "saved_config.yaml"
    save_config(cfg, temp_cfg_path)
    assert temp_cfg_path.is_file()
    reloaded = load_config(temp_cfg_path)
    assert reloaded["project"]["name"] == "Opportunity_HAR"
