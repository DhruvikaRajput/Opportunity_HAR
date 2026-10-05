"""Anatomical body-region representation and sensor grouping for OPPORTUNITY HAR.

Categorizes verified on-body sensor channels into 5 anatomical body regions:
    0: Trunk
    1: Right Arm
    2: Left Arm
    3: Right Leg
    4: Left Leg

Ambient and object sensors are kept strictly separate.
All mappings are grounded directly in the official column_names.txt documentation.
"""

from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# 5 Anatomical Region Names
BODY_REGIONS = ["trunk", "right_arm", "left_arm", "right_leg", "left_leg"]

# Exact 0-indexed column assignments based on column_names.txt
ANATOMICAL_SENSOR_COLUMNS: Dict[str, List[int]] = {
    # Trunk: BACK custom acc (16,17,18), HIP custom acc (4,5,6), BACK IMU (37-49: acc, gyro, mag, quat)
    "trunk": [4, 5, 6, 16, 17, 18] + list(range(37, 50)),
    
    # Right Arm: RUA_ (10,11,12), RWR (22,23,24), RUA^ (25,26,27), RH (34,35,36),
    # RUA IMU (50-62), RLA IMU (63-75)
    "right_arm": [10, 11, 12, 22, 23, 24, 25, 26, 27, 34, 35, 36] + list(range(50, 63)) + list(range(63, 76)),
    
    # Left Arm: LUA^ (7,8,9), LH (13,14,15), LUA_ (28,29,30), LWR (31,32,33),
    # LUA IMU (76-88), LLA IMU (89-101)
    "left_arm": [7, 8, 9, 13, 14, 15, 28, 29, 30, 31, 32, 33] + list(range(76, 89)) + list(range(89, 102)),
    
    # Right Leg: RKN^ (1,2,3), RKN_ (19,20,21), R-SHOE (118-133)
    "right_leg": [1, 2, 3, 19, 20, 21] + list(range(118, 134)),
    
    # Left Leg: L-SHOE (102-117)
    "left_leg": list(range(102, 118)),
}

# Anatomical Fixed Skeletal Adjacency:
# Limbs connect naturally to the central torso/trunk.
# Edges: (Trunk, Right Arm), (Trunk, Left Arm), (Trunk, Right Leg), (Trunk, Left Leg)
FIXED_ANATOMICAL_EDGES: List[Tuple[int, int]] = [
    (0, 1), (1, 0),  # Trunk <-> Right Arm
    (0, 2), (2, 0),  # Trunk <-> Left Arm
    (0, 3), (3, 0),  # Trunk <-> Right Leg
    (0, 4), (4, 0),  # Trunk <-> Left Leg
    (0, 0), (1, 1), (2, 2), (3, 3), (4, 4),  # Self-loops
]


def get_body_region_indices() -> Dict[str, List[int]]:
    """Return dictionary of 0-indexed column IDs for each anatomical region."""
    return ANATOMICAL_SENSOR_COLUMNS


def get_on_body_column_indices() -> List[int]:
    """Return sorted list of all 133 on-body sensor column indices."""
    indices = []
    for cols in ANATOMICAL_SENSOR_COLUMNS.values():
        indices.extend(cols)
    return sorted(indices)


def get_fixed_adjacency_matrix() -> np.ndarray:
    """Construct 5x5 fixed anatomical binary adjacency matrix.

    Nodes:
        0: Trunk, 1: Right Arm, 2: Left Arm, 3: Right Leg, 4: Left Leg

    Returns:
        np.ndarray: (5, 5) symmetric binary adjacency matrix with self-loops.
    """
    adj = np.zeros((5, 5), dtype=np.float32)
    for u, v in FIXED_ANATOMICAL_EDGES:
        adj[u, v] = 1.0
    return adj


def get_regional_channel_slices() -> Dict[str, Tuple[int, int]]:
    """Return slice offsets when on-body channels are concatenated region-by-region."""
    slices = {}
    current = 0
    for region in BODY_REGIONS:
        count = len(ANATOMICAL_SENSOR_COLUMNS[region])
        slices[region] = (current, current + count)
        current += count
    return slices


def plot_body_region_graph(save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """Visualize the anatomical body-region topology graph."""
    G = nx.Graph()
    labels = {
        0: "Trunk\n(22 ch)",
        1: "Right Arm\n(38 ch)",
        2: "Left Arm\n(38 ch)",
        3: "Right Leg\n(22 ch)",
        4: "Left Leg\n(16 ch)",
    }
    # Anatomical spatial layout
    pos = {
        0: (0.0, 0.5),    # Trunk
        1: (0.8, 0.6),    # Right Arm
        2: (-0.8, 0.6),   # Left Arm
        3: (0.5, -0.6),   # Right Leg
        4: (-0.5, -0.6),  # Left Leg
    }

    for u, v in FIXED_ANATOMICAL_EDGES:
        if u != v:
            G.add_edge(u, v)

    fig, ax = plt.subplots(figsize=(7, 7))
    nx.draw_networkx_nodes(G, pos, node_color="#4A90E2", node_size=3200, ax=ax, alpha=0.9)
    nx.draw_networkx_edges(G, pos, edge_color="#1D2A44", width=3.0, ax=ax)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_weight="bold", font_color="white", ax=ax)

    ax.set_title("OPPORTUNITY: Fixed Anatomical Sensor-Region Topology", fontsize=12, pad=15)
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig


if __name__ == "__main__":
    out_img = Path("results/dataset_inspection/body_region_mapping.png")
    plot_body_region_graph(out_img)
    print("Saved anatomical body-region graph to:", out_img)
    for reg, cols in ANATOMICAL_SENSOR_COLUMNS.items():
        print(f"  {reg:12s}: {len(cols):2d} sensor channels")
    print(f"  Total on-body : {len(get_on_body_column_indices())} channels")
