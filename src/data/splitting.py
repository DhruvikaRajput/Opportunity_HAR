"""Subject-based data splitting to prevent cross-subject data leakage.

Ensures that sensor recordings from the same individual subject never appear
concurrently in both training and test/validation sets.
"""

from typing import List, Tuple, Dict, Generator, Optional
import pandas as pd


def subject_split(
    df: pd.DataFrame,
    subject_column: str,
    train_subjects: List[str],
    val_subjects: Optional[List[str]] = None,
    test_subjects: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Partition a dataset strictly by subject identifiers to guarantee zero subject leakage.

    Args:
        df (pd.DataFrame): Input time-series dataset.
        subject_column (str): Name of the column containing subject IDs.
        train_subjects (List[str]): Subject IDs designated for training.
        val_subjects (Optional[List[str]]): Subject IDs designated for validation.
        test_subjects (Optional[List[str]]): Subject IDs designated for testing.

    Returns:
        Dict[str, pd.DataFrame]: Subsets with keys 'train', and optionally 'val' and 'test'.

    Raises:
        ValueError: If any subject ID overlaps between partitions.
    """
    train_set = set(train_subjects)
    val_set = set(val_subjects) if val_subjects else set()
    test_set = set(test_subjects) if test_subjects else set()

    # Verify partition exclusivity
    if train_set.intersection(val_set):
        raise ValueError(f"Subject leakage detected: train and val share {train_set.intersection(val_set)}")
    if train_set.intersection(test_set):
        raise ValueError(f"Subject leakage detected: train and test share {train_set.intersection(test_set)}")
    if val_set.intersection(test_set):
        raise ValueError(f"Subject leakage detected: val and test share {val_set.intersection(test_set)}")

    partitions = {
        "train": df[df[subject_column].isin(train_subjects)].copy()
    }
    if val_subjects:
        partitions["val"] = df[df[subject_column].isin(val_subjects)].copy()
    if test_subjects:
        partitions["test"] = df[df[subject_column].isin(test_subjects)].copy()

    return partitions


def generate_loso_splits(
    subjects: List[str],
    val_strategy: str = "none",
) -> Generator[Dict[str, List[str]], None, None]:
    """Generate Leave-One-Subject-Out (LOSO) partition combinations.

    Args:
        subjects (List[str]): Complete list of unique subject IDs.
        val_strategy (str): 'none' (train on N-1, test on 1), or
            'previous' (test on i, val on i-1, train on remainder).

    Yields:
        Dict[str, List[str]]: Mapping with keys 'test_subject', 'train_subjects',
            and optionally 'val_subjects'.
    """
    unique_subs = list(dict.fromkeys(subjects))
    n = len(unique_subs)
    if n < 2:
        raise ValueError(f"LOSO cross-validation requires at least 2 subjects; received {n}.")

    for i, test_sub in enumerate(unique_subs):
        remaining = [s for s in unique_subs if s != test_sub]
        if val_strategy == "previous" and len(remaining) >= 2:
            val_sub = remaining[-1]
            train_subs = remaining[:-1]
            yield {
                "test_subject": test_sub,
                "train_subjects": train_subs,
                "val_subjects": [val_sub],
            }
        else:
            yield {
                "test_subject": test_sub,
                "train_subjects": remaining,
                "val_subjects": [],
            }
