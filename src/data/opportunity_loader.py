"""Data loader for the official UCI OPPORTUNITY Activity Recognition benchmark.

Reads raw .dat files, extracts on-body sensor streams, parses ground-truth labels
(Locomotion or Gestures/ML_Both_Arms), handles missing values, and maintains subject/run metadata.

IMPORTANT RESEARCH RULE:
    Raw dataset files are strictly read-only and never modified.
"""

from typing import List, Dict, Tuple, Optional, Union, Any
from pathlib import Path
import json
import pandas as pd
import numpy as np

from src.utils.logging import get_logger

logger = get_logger("opportunity_loader")


class OpportunityLoader:
    """Loader and parser for the UCI OPPORTUNITY dataset files."""

    def __init__(
        self,
        dataset_dir: Optional[Union[str, Path]] = None,
        target_track: str = "Locomotion",  # 'Locomotion' or 'ML_Both_Arms'
        sensor_selection: str = "on_body",  # 'on_body', 'all_sensors', or list of column indices
    ):
        """Initialize OpportunityLoader.

        Args:
            dataset_dir (Optional[Union[str, Path]]): Directory containing .dat files.
                If None, automatically searches under data/raw/.
            target_track (str): Target classification task ('Locomotion' or 'ML_Both_Arms').
            sensor_selection (str): 'on_body' (133 wearable channels) or 'all_sensors'.
        """
        # Repository root anchor
        self.repo_root = Path(__file__).resolve().parent.parent.parent

        self.dataset_dir = self._resolve_dataset_dir(dataset_dir)
        self.target_track = target_track
        self.sensor_selection = sensor_selection

        # Load sensor and activity metadata using anchored repo root
        self.sensor_meta_path = self.repo_root / "data" / "processed" / "sensor_metadata.json"
        self.activity_meta_path = self.repo_root / "data" / "processed" / "activity_metadata.json"

        self.sensor_metadata = self._load_json(self.sensor_meta_path)
        self.activity_metadata = self._load_json(self.activity_meta_path)

        # Resolve target label column index
        if self.target_track == "Locomotion":
            self.label_column_idx = 243  # 0-indexed column 243 (1-indexed 244)
        elif self.target_track in ["ML_Both_Arms", "gestures"]:
            self.label_column_idx = 249  # 0-indexed column 249 (1-indexed 250)
        else:
            raise ValueError(f"Unknown target track: '{target_track}'. Choose 'Locomotion' or 'ML_Both_Arms'.")

        # Resolve sensor column indices
        self.sensor_indices = self._resolve_sensor_indices()
        logger.info(
            "OpportunityLoader initialized: %d sensor channels, target track '%s' (col %d)",
            len(self.sensor_indices),
            self.target_track,
            self.label_column_idx,
        )

    def _resolve_dataset_dir(self, dataset_dir: Optional[Union[str, Path]]) -> Path:
        import os
        # 1. Explicit path passed
        if dataset_dir is not None:
            p = Path(dataset_dir)
            if p.exists():
                return p

        # 2. Environment variable
        if "OPPORTUNITY_DATA_DIR" in os.environ:
            p = Path(os.environ["OPPORTUNITY_DATA_DIR"])
            if p.exists():
                return p

        # 3. Standard repository root data/raw
        repo_raw = self.repo_root / "data" / "raw"
        if repo_raw.exists():
            candidates = list(repo_raw.rglob("column_names.txt"))
            if candidates:
                return candidates[0].parent
            dat_candidates = list(repo_raw.rglob("S1-ADL1.dat"))
            if dat_candidates:
                return dat_candidates[0].parent

        # 4. Search common Google Drive paths if in Colab
        drive_candidates = [
            Path("/content/drive/MyDrive/Opportunity_HAR/data/raw"),
            Path("/content/drive/MyDrive/OpportunityUCIDataset/dataset"),
            Path("/content/drive/MyDrive/Opportunity/dataset"),
            Path("/content/drive/MyDrive/Opportunity"),
        ]
        for dp in drive_candidates:
            if dp.exists():
                dats = list(dp.rglob("S1-ADL1.dat"))
                if dats:
                    return dats[0].parent
                if (dp / "S1-ADL1.dat").exists():
                    return dp

        # 5. Check relative path as fallback
        rel_raw = Path("data/raw")
        if rel_raw.exists():
            candidates = list(rel_raw.rglob("column_names.txt"))
            if candidates:
                return candidates[0].parent
            dat_candidates = list(rel_raw.rglob("S1-ADL1.dat"))
            if dat_candidates:
                return dat_candidates[0].parent

        raise FileNotFoundError(
            "OPPORTUNITY dataset not found.\n"
            "Please ensure the 24 .dat files are placed in 'data/raw/' or in Google Drive ('/content/drive/MyDrive/...'),\n"
            "or set the environment variable OPPORTUNITY_DATA_DIR."
        )

    def _load_json(self, path: Path) -> dict:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _resolve_sensor_indices(self) -> List[int]:
        if isinstance(self.sensor_selection, list):
            return self.sensor_selection
        if self.sensor_metadata and "on_body_sensor_indices" in self.sensor_metadata:
            if self.sensor_selection == "on_body":
                return self.sensor_metadata["on_body_sensor_indices"]

        # Fallback by parsing column range if metadata file not generated yet
        # Columns 1 to 134 are on-body IMUs and accelerometers
        if self.sensor_selection == "on_body":
            return list(range(1, 134))
        # All sensors (excluding timestamp col 0 and labels 243-249)
        return list(range(1, 243))

    def load_recording(
        self,
        subject: str,
        run: str,
        interpolate_nans: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load a single recording file (e.g. S1-ADL1.dat).

        Args:
            subject (str): Subject ID (e.g. 'S1', 'S2', 'S3', 'S4').
            run (str): Run ID (e.g. 'ADL1', 'ADL2', 'ADL3', 'ADL4', 'ADL5', 'Drill').
            interpolate_nans (bool): Apply linear interpolation for missing sensor readings.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - sensor_data: Array of shape (T_timesteps, num_sensors), float32.
                - labels: Array of shape (T_timesteps,), int64.
                - timestamps: Array of shape (T_timesteps,), int64 (milliseconds).
        """
        filename = f"{subject}-{run}.dat"
        file_path = self.dataset_dir / filename
        if not file_path.is_file():
            raise FileNotFoundError(f"Recording file not found: {file_path}")

        # OPPORTUNITY dat files are whitespace separated
        raw_df = pd.read_csv(file_path, sep=r"\s+", header=None)

        if raw_df.shape[1] != 250:
            raise ValueError(f"Expected 250 columns in {filename}, found {raw_df.shape[1]}")

        timestamps = raw_df.iloc[:, 0].to_numpy(dtype=np.int64)
        labels = raw_df.iloc[:, self.label_column_idx].to_numpy(dtype=np.int64)
        sensor_df = raw_df.iloc[:, self.sensor_indices].copy()

        if interpolate_nans:
            # Linear interpolation along sensor streams, boundary fill with 0
            sensor_df = sensor_df.interpolate(method="linear", limit_direction="both").fillna(0.0)

        sensor_data = sensor_df.to_numpy(dtype=np.float32)
        return sensor_data, labels, timestamps

    def load_subject_recordings(
        self,
        subjects: List[str],
        runs: Optional[List[str]] = None,
        interpolate_nans: bool = True,
    ) -> List[Dict[str, Any]]:
        """Load multiple recordings grouped by subject and run.

        Args:
            subjects (List[str]): List of subjects e.g. ['S1', 'S2'].
            runs (Optional[List[str]]): List of runs e.g. ['ADL1', 'ADL2']. If None, loads all 6 runs.
            interpolate_nans (bool): Whether to interpolate NaNs.

        Returns:
            List[Dict[str, Any]]: List of recording dicts with keys:
                'subject', 'run', 'data', 'labels', 'timestamps'.
        """
        if runs is None:
            runs = ["ADL1", "ADL2", "ADL3", "ADL4", "ADL5", "Drill"]

        recordings = []
        for sub in subjects:
            for r in runs:
                try:
                    data, lbls, t_stamps = self.load_recording(sub, r, interpolate_nans=interpolate_nans)
                    recordings.append({
                        "subject": sub,
                        "run": r,
                        "data": data,
                        "labels": lbls,
                        "timestamps": t_stamps,
                    })
                    logger.info("Loaded recording: %s-%s (%d time-steps)", sub, r, len(data))
                except FileNotFoundError:
                    logger.warning("Recording %s-%s not found, skipping.", sub, r)

        return recordings
