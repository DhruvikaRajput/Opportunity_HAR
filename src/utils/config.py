"""Configuration loading and validation utilities.

Handles reading YAML configuration files and converting them to nested dictionaries
or namespaces for structured experiment management.
"""

from typing import Any, Dict
from pathlib import Path
import yaml


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load configuration from a YAML file.

    Args:
        config_path (str | Path): Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If parsing fails.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config or {}


def save_config(config: Dict[str, Any], output_path: str | Path) -> None:
    """Save configuration dictionary to a YAML file.

    Args:
        config (Dict[str, Any]): Configuration dictionary to persist.
        output_path (str | Path): Destination file path.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
