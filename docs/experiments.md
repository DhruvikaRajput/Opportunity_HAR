# Experiments & Empirical Baseline Results

This document summarizes the experiments executed on the real OPPORTUNITY dataset.

---

## 1. Classical Machine Learning Baselines (CPU)

- **Input**: Statistical features (mean, std, min, max) extracted across 133 on-body channels ($133 \times 4 = 532$ features per window).
- **Split**: Standard OPPORTUNITY train files (S1 all runs, S2 ADL1-3/Drill, S3 ADL1-3/Drill) vs Test (S3 ADL4-5).
- **Results**:

| Model | Accuracy | Macro F1 | Weighted F1 | Macro Precision | Macro Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **67.66%** | **54.62%** | **63.74%** | **60.06%** | **55.57%** |
| **Logistic Regression** | 51.95% | 45.49% | 52.37% | 60.07% | 45.74% |
| **Linear SVM** | 43.47% | 45.85% | 47.02% | 58.69% | 42.32% |

---

## 2. Anatomical Body-Region Ablation Study (CPU)

- **Objective**: Determine the contribution of each anatomical body region to Locomotion recognition.
- **Model**: Random Forest (100 estimators, fixed seed).
- **Results**:

| Ablation Scenario | Channels Kept | Accuracy | Macro F1 | Weighted F1 |
| :--- | :---: | :---: | :---: | :---: |
| **All Sensors (Clean)** | **133** | **67.66%** | **54.62%** | **63.74%** |
| **Without Trunk** | 114 | 67.78% | 54.99% | 64.20% |
| **Without Right Arm** | 95 | 66.64% | 53.63% | 62.75% |
| **Without Left Arm** | 95 | 66.78% | 53.52% | 63.16% |
| **Without Right Leg** | 111 | 67.94% | 54.72% | 64.62% |
| **Without Left Leg** | 117 | 66.25% | 53.03% | 62.03% |

**Key Finding**:
Removing arm sensors reduces Macro F1 by ~1.1%, while removing left-leg sensors causes the largest single drop (-1.59% Macro F1), highlighting the importance of lower-limb and coordination kinematics.

---

## 3. Sensor Failure Robustness Study (CPU)

- **Objective**: Measure degradation under missing sensor readings during inference.
- **Results**:

| Failure Scenario | Dropped Channels | Accuracy | Macro F1 | Degradation |
| :--- | :---: | :---: | :---: | :---: |
| **0% Failure (Clean)** | 0 | 67.66% | 54.62% | 0.00% |
| **10% Random Dropout** | 13 | 66.58% | 53.69% | -0.92% |
| **25% Random Dropout** | 33 | 60.63% | 46.56% | -8.05% |
| **50% Random Dropout** | 66 | 53.95% | 37.21% | -17.41% |
| **75% Random Dropout** | 99 | 35.80% | 10.54% | -44.07% |
| **Complete Right Leg Outage** | 22 | 41.90% | 20.77% | **-33.85%** |
| **Complete Left Arm Outage** | 38 | 60.07% | 46.03% | -8.59% |

**Key Finding**:
The system exhibits resilience up to 10% random sensor failure, but catastrophic degradation occurs when the right leg sensor cluster completely fails (-33.85% Macro F1 drop).

---

## 4. Leave-One-Subject-Out (LOSO) Cross-Validation (CPU)

- **Objective**: Evaluate cross-subject generalization across all four OPPORTUNITY subjects (S1, S2, S3, S4).
- **Protocol**: 4-fold cross-validation. In fold $i$, train strictly on 3 subjects and evaluate on held-out subject $i$. Scalers fit on training subjects only.
- **Model**: Random Forest.
- **Results**:

| Held-Out Subject | Accuracy | Macro F1 | Weighted F1 | Macro Precision | Macro Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **S1** | 77.96% | 80.46% | 77.55% | 80.21% | 82.60% |
| **S2** | 76.21% | 75.89% | 76.15% | 81.78% | 72.56% |
| **S3** | 69.70% | 53.62% | 66.57% | 57.51% | 53.68% |
| **S4** | 79.09% | 79.31% | 78.76% | 85.22% | 78.46% |
| **Mean** | **75.74%** | **72.32%** | **74.76%** | **76.18%** | **71.82%** |
| **Std Dev** | ±4.19% | ±12.62% | ±5.56% | ±12.62% | ±12.78% |

**Key Finding**:
Performance on S1, S2, and S4 is strong and consistent (~76-80% Macro F1), while S3 shows greater variability (53.62% Macro F1), indicating inter-subject kinematic differences during locomotion transitions.

---

## 5. Deep Learning & Graph Models (Prepared for GPU Training)

Per the project compute rules, full neural training is reserved for GPU environments:
- **1D CNN**: `python -m src.training.train_cnn --config configs/cnn.yaml`
- **Fixed Anatomical GNN**: `python -m src.training.train_fixed_gnn --config configs/fixed_gnn.yaml`
- **Activity-Adaptive GNN**: `python -m src.training.train_adaptive_gnn --config configs/adaptive_gnn.yaml`
- **Colab Notebook**: `notebooks/02_gpu_training.ipynb`
