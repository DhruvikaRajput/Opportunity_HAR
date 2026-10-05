"""Fixed Anatomical Graph Neural Network (Fixed GNN) for OPPORTUNITY HAR.

MATHEMATICAL ARCHITECTURE & TENSOR SHAPES:
=============================================================================
1. Anatomical Nodes (N = 5 Body Regions):
   - Node 0: Trunk      (19 sensor channels)
   - Node 1: Right Arm  (38 sensor channels)
   - Node 2: Left Arm   (38 sensor channels)
   - Node 3: Right Leg  (22 sensor channels)
   - Node 4: Left Leg   (16 sensor channels)

2. Fixed Anatomical Adjacency Matrix (A in R^{5x5}):
   Skeletal connectivity where limbs attach mechanically to the central trunk:
   Edges: (Trunk, Right Arm), (Trunk, Left Arm), (Trunk, Right Leg), (Trunk, Left Leg)
   Symmetric normalized adjacency with self-loops:
   \tilde{A} = \tilde{D}^{-1/2} (A + I) \tilde{D}^{-1/2}

3. Regional Temporal Encoders:
   For each region i with C_i channels and temporal window T:
   h_i = Conv1D(C_i, D, kernel=5) -> BatchNorm1D -> ReLU -> AdaptiveAvgPool1D(1)
   Node feature matrix: H^{(0)} in R^{B x N x D}  (B=Batch, N=5, D=Node Feature Dim)

4. Graph Message Passing (GCN Layer):
   H^{(1)} = ReLU( \tilde{A} H^{(0)} W_1 )
   H^{(2)} = ReLU( \tilde{A} H^{(1)} W_2 )

5. Graph Readout & Classification:
   Readout = Concat[ MeanPool(H^{(2)}), MaxPool(H^{(2)}) ] in R^{B x 2D}
   Logits  = Linear(2D, num_classes) in R^{B x K}
=============================================================================
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn

from src.features.body_regions import (
    BODY_REGIONS,
    ANATOMICAL_SENSOR_COLUMNS,
    FIXED_ANATOMICAL_EDGES,
    get_on_body_column_indices,
    get_fixed_adjacency_matrix,
)


class RegionalTemporalEncoder(nn.Module):
    """Encodes continuous multi-channel sensor signals of one body region into a node embedding."""

    def __init__(self, in_channels: int, node_dim: int = 64, kernel_size: int = 5, dropout: float = 0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, node_dim, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(node_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Conv1d(node_dim, node_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(node_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): (B, C_region, T)

        Returns:
            torch.Tensor: (B, node_dim)
        """
        feat = self.encoder(x)  # (B, D, 1)
        return feat.squeeze(-1)  # (B, D)


class GraphConvolutionLayer(nn.Module):
    """Standard Graph Convolutional Network (GCN) layer operating on batch graph tensors."""

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.2):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.batch_norm = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, h: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            h (torch.Tensor): Node features of shape (B, N, in_features).
            adj_norm (torch.Tensor): Normalized adjacency matrix of shape (N, N) or (B, N, N).

        Returns:
            torch.Tensor: Updated node representations of shape (B, N, out_features).
        """
        # Linear transformation: (B, N, in_feat) -> (B, N, out_feat)
        support = self.linear(h)

        # Message passing: (N, N) x (B, N, out_feat) or (B, N, N) x (B, N, out_feat)
        if adj_norm.dim() == 2:
            out = torch.einsum("ij,bjk->bik", adj_norm, support)
        else:
            out = torch.bmm(adj_norm, support)

        # Reshape for BatchNorm1d: (B, out_feat, N)
        B, N, D = out.shape
        out = out.transpose(1, 2)
        out = self.batch_norm(out)
        out = out.transpose(1, 2)
        out = self.relu(out)
        out = self.dropout(out)
        return out


class FixedAnatomicalGNN(nn.Module):
    """Fixed Anatomical Graph Neural Network for Human Activity Recognition."""

    def __init__(
        self,
        num_classes: int = 5,
        sequence_length: int = 30,
        node_dim: int = 64,
        gcn_hidden_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.node_dim = node_dim
        self.num_nodes = len(BODY_REGIONS)

        # Map channel indices for each region from the full 133 on-body vector
        sorted_cols = get_on_body_column_indices()
        col_to_pos = {c: idx for idx, c in enumerate(sorted_cols)}

        self.region_channel_indices: Dict[str, List[int]] = {}
        self.encoders = nn.ModuleDict()

        for region in BODY_REGIONS:
            raw_cols = ANATOMICAL_SENSOR_COLUMNS[region]
            pos_indices = [col_to_pos[c] for c in raw_cols]
            self.region_channel_indices[region] = pos_indices
            self.encoders[region] = RegionalTemporalEncoder(
                in_channels=len(pos_indices),
                node_dim=node_dim,
                dropout=dropout,
            )

        # Compute normalized fixed anatomical adjacency: \tilde{A} = \tilde{D}^{-1/2} A_{self} \tilde{D}^{-1/2}
        raw_adj = get_fixed_adjacency_matrix()  # (5, 5) with self loops
        d_vec = np.sum(raw_adj, axis=1)
        d_inv_sqrt = np.power(d_vec, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        norm_adj = d_mat_inv_sqrt @ raw_adj @ d_mat_inv_sqrt

        self.register_buffer("fixed_adj_norm", torch.from_numpy(norm_adj).float())

        # GCN Message Passing Layers
        self.gcn1 = GraphConvolutionLayer(node_dim, gcn_hidden_dim, dropout=dropout)
        self.gcn2 = GraphConvolutionLayer(gcn_hidden_dim, gcn_hidden_dim, dropout=dropout)

        # Readout Classifier (concatenates mean and max node embeddings)
        self.classifier = nn.Sequential(
            nn.Linear(gcn_hidden_dim * 2, gcn_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(gcn_hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Full sensor window tensor of shape (B, 133, T).

        Returns:
            torch.Tensor: Unnormalized activity logits of shape (B, num_classes).
        """
        B, C, T = x.shape
        node_embeddings = []

        # 1. Regional Temporal Encoding: (B, C_reg, T) -> (B, node_dim)
        for region in BODY_REGIONS:
            indices = self.region_channel_indices[region]
            x_region = x[:, indices, :]
            emb = self.encoders[region](x_region)  # (B, D)
            node_embeddings.append(emb)

        # Stack into graph node matrix: (B, N=5, D)
        h0 = torch.stack(node_embeddings, dim=1)

        # 2. Fixed Anatomical Message Passing
        h1 = self.gcn1(h0, self.fixed_adj_norm)  # (B, 5, D)
        h2 = self.gcn2(h1, self.fixed_adj_norm)  # (B, 5, D)

        # 3. Graph Readout (Mean-pool + Max-pool over 5 body nodes)
        h_mean = torch.mean(h2, dim=1)  # (B, D)
        h_max, _ = torch.max(h2, dim=1)  # (B, D)
        graph_repr = torch.cat([h_mean, h_max], dim=-1)  # (B, 2D)

        # 4. Classification Head
        logits = self.classifier(graph_repr)  # (B, num_classes)
        return logits
