# Model Architectures: Fixed and Activity-Adaptive Anatomical GNNs

This document describes the mathematical formulation, tensor shapes, and architectural design of the models developed for the research hypothesis.

---

## 1. Baseline 1D CNN (`src/models/cnn.py`)

- **Input Shape**: $(B, C, T) = (B, 133, 30)$
- **Layers**:
  - Block 1: $\text{Conv1D}(133, 64, \text{kernel}=5) \to \text{BatchNorm1d} \to \text{ReLU} \to \text{MaxPool1d}(2) \to \text{Dropout}(0.2)$
  - Block 2: $\text{Conv1D}(64, 128, \text{kernel}=5) \to \text{BatchNorm1d} \to \text{ReLU} \to \text{MaxPool1d}(2) \to \text{Dropout}(0.2)$
  - Block 3: $\text{Conv1D}(128, 128, \text{kernel}=3) \to \text{BatchNorm1d} \to \text{ReLU} \to \text{AdaptiveAvgPool1d}(1)$
  - Readout: $\text{Flatten} \to \text{Linear}(128, K)$
- **Output Shape**: $(B, K)$ where $K=5$ activity logits.

---

## 2. Fixed Anatomical GNN (`src/models/fixed_gnn.py`)

### Nodes (5 Anatomical Body Regions)
1. **Trunk** (19 channels: BACK IMU + Back custom acc + Hip custom acc)
2. **Right Arm** (38 channels: RUA IMU + RLA IMU + RUA^ + RUA_ + RWR + RH)
3. **Left Arm** (38 channels: LUA IMU + LLA IMU + LUA^ + LUA_ + LWR + LH)
4. **Right Leg** (22 channels: R-SHOE + RKN^ + RKN_)
5. **Left Leg** (16 channels: L-SHOE)

### Regional Temporal Encoding
For region $i$ with $C_i$ channels:
$$h_i = \text{AdaptiveAvgPool1d}(\text{ReLU}(\text{BatchNorm}(\text{Conv1D}(C_i, D, 5)))) \in \mathbb{R}^D$$
Stacking all 5 regions yields $H^{(0)} \in \mathbb{R}^{B \times 5 \times D}$ (with $D=64$).

### Fixed Anatomical Skeletal Adjacency
Symmetric normalized matrix with self loops:
$$\tilde{A} = \tilde{D}^{-1/2} (A + I) \tilde{D}^{-1/2} \in \mathbb{R}^{5 \times 5}$$
Where edges link limbs directly to the central trunk:
$$\mathcal{E} = \{(\text{Trunk}, \text{Right Arm}), (\text{Trunk}, \text{Left Arm}), (\text{Trunk}, \text{Right Leg}), (\text{Trunk}, \text{Left Leg})\}$$

### Message Passing & Readout
$$H^{(l+1)} = \text{ReLU}(\text{BatchNorm}(\tilde{A} H^{(l)} W_l))$$
$$\text{Logits} = \text{Linear}([\text{MeanPool}(H^{(2)}) \parallel \text{MaxPool}(H^{(2)})])$$

---

## 3. Activity-Adaptive Anatomical GNN (`src/models/adaptive_gnn.py`)

### The Research Question
*Can allowing relationships between anatomical body regions to dynamically adapt to the input/activity state improve recognition and robustness compared to a static skeletal graph?*

### Relation Generator Formulation
Given regional node representations $h_i, h_j \in \mathbb{R}^D$:
1. Pairwise attention scoring:
   $$e_{ij} = \text{LeakyReLU}(v^T [W_q h_i \parallel W_k h_j])$$
2. Softmax normalization across destination nodes:
   $$A_{\text{adapt}}(x)_{ij} = \frac{\exp(e_{ij})}{\sum_{m=1}^5 \exp(e_{im})}$$
   Yields dynamic matrix $A_{\text{adapt}}(x) \in \mathbb{R}^{B \times 5 \times 5}$.

3. Fusing with the Fixed Anatomical Prior:
   $$A_{\text{dyn}}(x) = (1 - \lambda) \tilde{A}_{\text{fixed}} + \lambda A_{\text{adapt}}(x)$$
   Where $\lambda = \text{sigmoid}(\gamma) \in (0, 1)$ is a learnable gating parameter.

4. Dynamic Message Passing:
   $$H^{(l+1)} = \text{ReLU}(\text{BatchNorm}(A_{\text{dyn}}(x) H^{(l)} W_l))$$

5. Research Inspection:
   The model returns $A_{\text{dyn}}(x)$ during evaluation to visualize the learned inter-region attention matrices for each activity class.
