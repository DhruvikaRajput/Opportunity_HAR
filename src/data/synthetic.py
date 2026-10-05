"""Synthetic multi-channel IMU time-series generator for testing and validation.

DISCLAIMER:
    SYNTHETIC TEST DATA — NOT OPPORTUNITY
    Do not use the synthetic results in any research paper or empirical claim.
    This module exists solely to verify software mechanics, gradient flow,
    and end-to-end pipeline integrity while real dataset files are unavailable.
"""

from typing import List, Tuple, Dict
import numpy as np
import pandas as pd


def generate_synthetic_imu_data(
    num_subjects: int = 3,
    num_channels: int = 18,
    samples_per_subject: int = 2400,
    num_classes: int = 5,
    sampling_rate_hz: float = 30.0,
    noise_std: float = 0.05,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, Dict]:
    """Generate synthetic continuous multi-channel IMU recordings with distinct class patterns.

    Args:
        num_subjects (int): Number of distinct synthetic subjects (e.g. S1, S2, S3).
        num_channels (int): Number of sensory channels (e.g. 18 channels).
        samples_per_subject (int): Number of sequential time samples per subject.
        num_classes (int): Number of activity classes (0 to num_classes - 1).
        sampling_rate_hz (float): Nominal sampling rate in Hz.
        noise_std (float): Standard deviation of Gaussian sensor noise.
        random_seed (int): Seed for reproducible data generation.

    Returns:
        Tuple[pd.DataFrame, Dict]:
            - pd.DataFrame: Tabular stream with columns ['subject_id', 'timestamp',
              'sensor_ch_00' ... 'sensor_ch_N-1', 'activity_label'].
            - Dict: Metadata describing the synthetic setup.
    """
    rng = np.random.RandomState(random_seed)
    channel_cols = [f"sensor_ch_{i:02d}" for i in range(num_channels)]
    records = []

    dt = 1.0 / sampling_rate_hz
    # Distinct characteristic frequencies and amplitudes per activity class
    base_freqs = np.linspace(0.8, 4.0, num_classes)
    base_amps = np.linspace(0.6, 2.2, num_classes)

    for subj_idx in range(1, num_subjects + 1):
        subj_name = f"Subject_{subj_idx:02d}"
        t = np.arange(samples_per_subject) * dt

        # Generate block activity sequences (e.g. 120 samples = 4 seconds per activity state)
        block_len = 120
        num_blocks = int(np.ceil(samples_per_subject / block_len))
        block_labels = rng.randint(0, num_classes, size=num_blocks)
        labels = np.repeat(block_labels, block_len)[:samples_per_subject]

        # Multi-channel sensor signals driven by current activity
        signals = np.zeros((samples_per_subject, num_channels), dtype=np.float32)

        # Generate smooth signals with class-specific amplitude and frequency
        for c in range(num_channels):
            phase = (c * np.pi) / num_channels
            # Channel modulation factor
            ch_factor = 0.8 + 0.4 * (c % 3)
            
            amp = base_amps[labels] * ch_factor
            freq = base_freqs[labels]
            
            # Harmonic pattern
            signal = amp * np.sin(2 * np.pi * freq * t + phase)
            # Add secondary harmonic
            signal += 0.3 * amp * np.cos(4 * np.pi * freq * t + phase)
            # Add class-dependent offset
            signal += (labels - (num_classes / 2.0)) * 0.25
            
            noise = rng.normal(0, noise_std, size=samples_per_subject)
            signals[:, c] = signal + noise

        df_subj = pd.DataFrame(signals, columns=channel_cols)
        df_subj.insert(0, "subject_id", subj_name)
        df_subj.insert(1, "timestamp", t)
        df_subj["activity_label"] = labels

        records.append(df_subj)

    df_all = pd.concat(records, ignore_index=True)

    metadata = {
        "dataset_type": "SYNTHETIC TEST DATA — NOT OPPORTUNITY",
        "num_subjects": num_subjects,
        "subjects": [f"Subject_{i:02d}" for i in range(1, num_subjects + 1)],
        "num_channels": num_channels,
        "channel_names": channel_cols,
        "num_classes": num_classes,
        "sampling_rate_hz": sampling_rate_hz,
        "total_samples": len(df_all),
    }

    return df_all, metadata
