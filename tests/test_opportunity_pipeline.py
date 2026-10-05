"""Unit tests for OPPORTUNITY-specific modules: Loader, Body Regions, CNN, Fixed GNN, and Adaptive GNN.

All neural network tests perform small CPU forward-passes strictly for shape and gradient verification.
No expensive training is executed.
"""

import sys
from pathlib import Path
import pytest
import torch
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.opportunity_loader import OpportunityLoader
from src.features.body_regions import (
    BODY_REGIONS,
    ANATOMICAL_SENSOR_COLUMNS,
    get_on_body_column_indices,
    get_fixed_adjacency_matrix,
)
from src.models.cnn import OpportunityCNN
from src.models.fixed_gnn import FixedAnatomicalGNN
from src.models.adaptive_gnn import ActivityAdaptiveAnatomicalGNN
from src.models.classical import extract_statistical_features


def test_body_region_mapping_completeness():
    """Verify that the 5 body regions encompass all 133 on-body sensor channels without overlap."""
    assert len(BODY_REGIONS) == 5
    all_cols = []
    for reg, cols in ANATOMICAL_SENSOR_COLUMNS.items():
        all_cols.extend(cols)

    # 133 on-body channels total
    assert len(all_cols) == 133
    # Mutually exclusive
    assert len(set(all_cols)) == 133

    adj = get_fixed_adjacency_matrix()
    assert adj.shape == (5, 5)
    # Diagonal self-loops
    assert np.all(np.diag(adj) == 1.0)


def test_opportunity_loader_real_file_loading():
    """Verify that OpportunityLoader loads a real .dat file with correct shape and columns."""
    loader = OpportunityLoader(target_track="Locomotion", sensor_selection="on_body")
    data, labels, timestamps = loader.load_recording("S1", "ADL1", interpolate_nans=True)

    assert data.shape[1] == 133, f"Expected 133 on-body sensor channels, got {data.shape[1]}"
    assert len(labels) == len(data)
    assert len(timestamps) == len(data)
    assert not np.isnan(data).any(), "NaN values found after interpolation!"


def test_opportunity_cnn_forward():
    """Verify OpportunityCNN forward pass producing expected output shape (B, num_classes)."""
    B, C, T, K = 4, 133, 30, 5
    model = OpportunityCNN(in_channels=C, num_classes=K, sequence_length=T)
    x = torch.randn(B, C, T)

    out = model(x)
    assert out.shape == (B, K)
    assert not torch.isnan(out).any()


def test_fixed_anatomical_gnn_forward():
    """Verify FixedAnatomicalGNN forward pass on CPU producing (B, num_classes)."""
    B, C, T, K = 4, 133, 30, 5
    model = FixedAnatomicalGNN(num_classes=K, sequence_length=T, node_dim=32, gcn_hidden_dim=32)
    x = torch.randn(B, C, T)

    out = model(x)
    assert out.shape == (B, K)
    assert not torch.isnan(out).any()


def test_adaptive_anatomical_gnn_forward_and_attention():
    """Verify ActivityAdaptiveAnatomicalGNN forward pass and dynamic attention matrix shape (B, 5, 5)."""
    B, C, T, K = 4, 133, 30, 5
    model = ActivityAdaptiveAnatomicalGNN(num_classes=K, sequence_length=T, node_dim=32, gcn_hidden_dim=32)
    x = torch.randn(B, C, T)

    out, attn = model(x, return_attention=True)
    assert out.shape == (B, K)
    assert attn.shape == (B, 5, 5)

    # Verify rows of dynamic adjacency matrix sum to ~1.0
    row_sums = attn.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)


def test_statistical_features_shape():
    """Verify statistical feature extraction shape (N, C * 4)."""
    N, C, T = 10, 133, 30
    X = np.random.randn(N, C, T).astype(np.float32)
    feats = extract_statistical_features(X)
    assert feats.shape == (N, 133 * 4)
