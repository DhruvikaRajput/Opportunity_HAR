"""Activity-Adaptive Anatomical Graph Neural Network (Adaptive GNN) for OPPORTUNITY HAR.

MATHEMATICAL ARCHITECTURE & RESEARCH DESIGN:
=============================================================================
1. Anatomical Nodes (N = 5 Body Regions):
   - Node 0: Trunk      (19 sensor channels)
   - Node 1: Right Arm  (38 sensor channels)
   - Node 2: Left Arm   (38 sensor channels)
   - Node 3: Right Leg  (22 sensor channels)
   - Node 4: Left Leg   (16 sensor channels)

2. Node Feature Encoding:
   Each region i encodes its temporal window into initial node representation:
   h_i in R^D  -->  H^{(0)} in R^{B x N x D}

3. Input-Adaptive Relation / Edge-Weight Generator:
   For each window sample, pairwise attention logits between regions i and j:
   e_{ij} = LeakyReLU( v^T [ W_q h_i || W_k h_j ] )
   Normalized across destination nodes via Softmax:
   A_{adapt}(x)_{ij} = exp(e_{ij}) / sum_{k=1}^N exp(e_{ik})
   Shape: A_{adapt}(x) in R^{B x N x N}  (Per-window, row-normalized to sum to 1)

4. Combining Adaptive Interactions with Anatomical Skeletal Prior:
   A_{dyn}(x) = (1 - lambda) * \tilde{A}_{fixed} + lambda * A_{adapt}(x)
   where lambda = sigmoid(gamma) in (0, 1) is a learnable balance coefficient.
   This ensures message passing dynamically reflects the activity context while
   grounded in anatomical skeletal constraints.

5. Dynamic Message Passing:
   H^{(1)} = ReLU( A_{dyn}(x) H^{(0)} W_1 )
   H^{(2)} = ReLU( A_{dyn}(x) H^{(1)} W_2 )

6. Learned Edge Inspection:
   The forward pass can optionally return A_{adapt}(x) and A_{dyn}(x) for
   scientific visualization and analysis across different activity classes.
=============================================================================
"""

from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import numpy as np

from src.features.body_regions import (
    BODY_REGIONS,
    ANATOMICAL_SENSOR_COLUMNS,
    get_on_body_column_indices,
    get_fixed_adjacency_matrix,
)
from src.models.fixed_gnn import RegionalTemporalEncoder


class AdaptiveRelationGenerator(nn.Module):
    """Generates an input-dependent, activity-adaptive adjacency matrix A(x) in R^{B x N x N}."""

    def __init__(self, node_dim: int = 64, num_nodes: int = 5):
        super().__init__()
        self.node_dim = node_dim
        self.num_nodes = num_nodes

        self.W_q = nn.Linear(node_dim, node_dim, bias=False)
        self.W_k = nn.Linear(node_dim, node_dim, bias=False)
        self.v = nn.Linear(node_dim * 2, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Compute input-dependent relational edge weights.

        Args:
            h (torch.Tensor): Node feature tensor of shape (B, N, D).

        Returns:
            torch.Tensor: Normalized attention adjacency matrix of shape (B, N, N).
        """
        B, N, D = h.shape

        # Query & Key representations: (B, N, D)
        q = self.W_q(h)
        k = self.W_k(h)

        # Pairwise concatenation: (B, N, N, 2D)
        q_exp = q.unsqueeze(2).expand(B, N, N, D)
        k_exp = k.unsqueeze(1).expand(B, N, N, D)
        pairs = torch.cat([q_exp, k_exp], dim=-1)  # (B, N, N, 2D)

        # Unnormalized edge scores: (B, N, N)
        scores = self.leaky_relu(self.v(pairs)).squeeze(-1)

        # Normalize along target node dimension
        a_adapt = self.softmax(scores)  # (B, N, N), row sums = 1
        return a_adapt


class AdaptiveGraphConvolutionLayer(nn.Module):
    """Dynamic Graph Convolution utilizing per-sample dynamic adjacency matrices."""

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.2):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.batch_norm = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, h: torch.Tensor, adj_dyn: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            h (torch.Tensor): (B, N, in_features).
            adj_dyn (torch.Tensor): Dynamic batch adjacency of shape (B, N, N).

        Returns:
            torch.Tensor: (B, N, out_features).
        """
        # Linear feature transformation: (B, N, in_feat) -> (B, N, out_feat)
        support = self.linear(h)
        # Message passing with batch matrix multiplication: (B, N, N) x (B, N, out_feat) -> (B, N, out_feat)
        out = torch.bmm(adj_dyn, support)

        # BatchNorm across node channels: (B, out_feat, N)
        out = out.transpose(1, 2)
        out = self.batch_norm(out)
        out = out.transpose(1, 2)
        out = self.relu(out)
        out = self.dropout(out)
        return out


class ActivityAdaptiveAnatomicalGNN(nn.Module):
    """Activity-Adaptive Anatomical Graph Neural Network for Human Activity Recognition."""

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

        # Fixed anatomical adjacency prior
        raw_adj = get_fixed_adjacency_matrix()
        d_vec = np.sum(raw_adj, axis=1, keepdims=True)
        norm_adj = raw_adj / np.maximum(d_vec, 1e-8)
        self.register_buffer("fixed_adj_norm", torch.from_numpy(norm_adj).float())

        # Adaptive Relation Generator
        self.relation_generator = AdaptiveRelationGenerator(node_dim=node_dim, num_nodes=self.num_nodes)

        # Learnable balance coefficient between fixed prior and adaptive relations
        # Starts initialized near 0.5
        self.lambda_param = nn.Parameter(torch.tensor(0.0))

        # Dynamic GCN Message Passing Layers
        self.gcn1 = AdaptiveGraphConvolutionLayer(node_dim, gcn_hidden_dim, dropout=dropout)
        self.gcn2 = AdaptiveGraphConvolutionLayer(gcn_hidden_dim, gcn_hidden_dim, dropout=dropout)

        # Readout Classifier
        self.classifier = nn.Sequential(
            nn.Linear(gcn_hidden_dim * 2, gcn_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(gcn_hidden_dim, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            x (torch.Tensor): (B, 133, T) sensor tensor.
            return_attention (bool): If True, returns learned dynamic adjacency matrix.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Logits of shape (B, num_classes).
                - A_dyn (Optional): Dynamic adjacency tensor of shape (B, 5, 5).
        """
        B, C, T = x.shape
        node_embeddings = []

        # 1. Regional Temporal Encoders: (B, C_reg, T) -> (B, D)
        for region in BODY_REGIONS:
            indices = self.region_channel_indices[region]
            x_region = x[:, indices, :]
            emb = self.encoders[region](x_region)
            node_embeddings.append(emb)

        h0 = torch.stack(node_embeddings, dim=1)  # (B, 5, D)

        # 2. Activity-Adaptive Edge Weight Generation
        a_adapt = self.relation_generator(h0)  # (B, 5, 5)

        # 3. Fuse with Anatomical Prior: A_dyn = (1 - lam)*A_fixed + lam*A_adapt
        lam = torch.sigmoid(self.lambda_param)
        a_fixed_batch = self.fixed_adj_norm.unsqueeze(0).expand(B, self.num_nodes, self.num_nodes)
        a_dyn = (1.0 - lam) * a_fixed_batch + lam * a_adapt
        a_dyn = a_dyn / (a_dyn.sum(dim=-1, keepdim=True) + 1e-8)

        # 4. Dynamic Message Passing
        h1 = self.gcn1(h0, a_dyn)
        h2 = self.gcn2(h1, a_dyn)

        # 5. Readout (Mean + Max pooling over 5 body nodes)
        h_mean = torch.mean(h2, dim=1)
        h_max, _ = torch.max(h2, dim=1)
        graph_repr = torch.cat([h_mean, h_max], dim=-1)

        # 6. Classification
        logits = self.classifier(graph_repr)

        if return_attention:
            return logits, a_dyn
        return logits
