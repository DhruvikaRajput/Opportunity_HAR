"""Hardware device utilities for PyTorch (CPU and CUDA GPU management).

Allows seamless switching and verification between CPU and CUDA devices,
with hardware logging capabilities.
"""

from typing import Optional
import logging
import torch

logger = logging.getLogger(__name__)


def get_device(preferred_device: Optional[str] = None) -> torch.device:
    """Select and return the computation device (CPU or CUDA GPU).

    Args:
        preferred_device (Optional[str]): Preferred device identifier, e.g., 'cuda',
            'cuda:0', 'cpu', or None. If None or 'auto', automatically detects CUDA.

    Returns:
        torch.device: The resolved PyTorch device.
    """
    if preferred_device is not None and preferred_device.lower() != "auto":
        requested = preferred_device.lower()
        if requested.startswith("cuda"):
            if torch.cuda.is_available():
                device = torch.device(requested)
            else:
                logger.warning(
                    "CUDA requested ('%s') but CUDA is not available. Falling back to CPU.",
                    requested,
                )
                device = torch.device("cpu")
        elif requested == "cpu":
            device = torch.device("cpu")
        else:
            logger.warning(
                "Unknown device requested: '%s'. Defaulting to CPU.",
                requested,
            )
            device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    return device


def print_device_info(device: Optional[torch.device] = None) -> dict:
    """Collect and print information about the active PyTorch computation device.

    Args:
        device (Optional[torch.device]): Specific device to query. Defaults to active device.

    Returns:
        dict: Diagnostic dictionary containing hardware details.
    """
    if device is None:
        device = get_device()

    info = {
        "device_type": device.type,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    if device.type == "cuda" and torch.cuda.is_available():
        cuda_idx = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(cuda_idx)
        info["device_name"] = props.name
        info["total_memory_gb"] = round(props.total_memory / (1024**3), 2)
        info["compute_capability"] = f"{props.major}.{props.minor}"
        logger.info(
            "Using CUDA GPU [%d]: %s (Memory: %.2f GB, Compute: %s)",
            cuda_idx,
            info["device_name"],
            info["total_memory_gb"],
            info["compute_capability"],
        )
    else:
        info["device_name"] = "CPU"
        logger.info("Using computation device: CPU")

    return info
