# Opportunity_HAR: Anatomical & Adaptive Graph Neural Networks for Human Activity Recognition

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![CI Tests](https://img.shields.io/badge/tests-34%20passed-brightgreen.svg)]()
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

A research-oriented, reproducible software framework for sensor-based Human Activity Recognition (HAR) on the **UCI OPPORTUNITY** benchmark dataset.

---

## 🔬 Research Focus & Central Hypothesis

Human body kinematics and physical activities inherently induce dynamic coordination among anatomical limbs and body regions. This project experimentally investigates:

> **Central Hypothesis**: Modeling topological relationships between anatomical sensor regions improves human activity recognition performance and noise/outage robustness compared with treating sensor channels independently. Furthermore, allowing inter-regional topological relationships to adapt dynamically based on the input activity state yields superior representation capacity over a static anatomical skeletal prior.

### Methodological Model Progression:
$$\text{Classical ML (RF / SVM)} \longrightarrow \text{1D CNN} \longrightarrow \text{Fixed Anatomical GNN} \longrightarrow \text{Activity-Adaptive Anatomical GNN}$$

---

## 📊 Dataset Profile & Verification (UCI OPPORTUNITY)

Derived strictly from the official documentation (`column_names.txt`, `label_legend.txt`) and physical verification of all 24 raw `.dat` recordings:

- **Recordings**: 24 `.dat` files (4 Subjects: `S1`, `S2`, `S3`, `S4`; 6 runs each: `ADL1` to `ADL5`, plus `Drill`).
- **Total Samples**: 869,387 time steps.
- **Sampling Frequency**: **30.0 Hz** (Column 0: `MILLISEC`, timestamp intervals $\Delta t \approx 33.33\text{ ms}$).
- **Total Columns**: 250 (1 timestamp column, 242 sensor columns, 7 annotation/label columns).
- **On-Body Wearable Sensors**: **133 channels** (Inertial Measurement Units: Custom Inertial, Sun SPOTs, XSens).
- **Target Label Tracks**:
  - **Locomotion** (Column 243): 5 states (Stand, Walk, Sit, Lie, Null).
  - **Gestures / ML_Both_Arms** (Column 249): 17 high-level activities of daily living.

### Verified Anatomical Body-Region Partitioning (133 channels):
| Anatomical Region | Sensor IDs & Types | Channels |
| :--- | :--- | :---: |
| **Trunk** | Back (ACC, GYRO, MAG), Sun SPOTs on Back | **19** |
| **Right Arm** | Upper/Lower Right Arm, Right Hand (Inertials & IMUs) | **38** |
| **Left Arm** | Upper/Lower Left Arm, Left Hand (Inertials & IMUs) | **38** |
| **Right Leg** | Right Lower/Upper Leg, Right Shoe (ACC & Inertials) | **22** |
| **Left Leg** | Left Lower/Upper Leg, Left Shoe (ACC & Inertials) | **16** |
| **Total** | *Mutually exclusive, complete coverage of on-body sensors* | **133** |

Machine-readable schemas:
- `data/processed/sensor_metadata.json`
- `data/processed/activity_metadata.json`

---

## ⚡ Compute Rules & Hardware Guard

To prevent unfeasible execution on laptop CPUs:
- **CPU Allowed**: Dataset inspection, verification, leak-free preprocessing, classical ML training, body-region ablation, sensor robustness experiments, Leave-One-Subject-Out (LOSO) cross-validation for classical models, unit testing, and minimal forward-pass sanity checks.
- **CUDA GPU Mandatory**: Full CNN, Fixed GNN, and Activity-Adaptive GNN training.
- **Strict Guard**: The training entry points inspect `torch.cuda.is_available()`. If CUDA is not detected, execution stops immediately with an informative message rather than silently hanging the laptop CPU on expensive backpropagation.

---

## 📁 Repository Architecture

```text
Opportunity_HAR/
├── configs/                  # Experiment configurations
│   ├── cnn.yaml              # 1D CNN training config
│   ├── fixed_gnn.yaml        # Fixed Anatomical GNN config
│   └── adaptive_gnn.yaml     # Activity-Adaptive GNN config
│
├── data/
│   ├── raw/                  # Official UCI OPPORTUNITY dataset (.dat, immutable)
│   └── processed/            # Preprocessed, windowed arrays (.npz) & metadata (.json)
│
├── docs/                     # Research documentation
│   ├── methodology.md        # Preprocessing, sliding window, and evaluation protocols
│   ├── model_architecture.md # Mathematical formulations of CNN, Fixed GNN, and Adaptive GNN
│   └── experiments.md        # Complete empirical baseline, ablation, robustness, and LOSO results
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb # Dataset inspection and distribution analysis
│   └── 02_gpu_training.ipynb        # Complete Google Colab GPU training workflow
│
├── results/                  # Persisted artifacts, metric tables, and figures
│   ├── dataset_inspection/   # Column distributions, body-region topology graph
│   ├── baseline/             # Classical ML confusion matrices and metrics
│   ├── body_ablation/        # 6-condition ablation comparison plots and tables
│   ├── robustness/           # Degradation curves under sensor drop and regional outage
│   └── loso/                 # 4-subject cross-validation results and figures
│
├── src/
│   ├── data/                 # Data loaders, preprocessing, windowing, and splitting
│   │   ├── opportunity_loader.py    # Robust .dat file loader with linear interpolation
│   │   ├── preprocessing.py         # Leak-free SensorStandardScaler
│   │   ├── windowing.py             # Configurable sliding-window generator
│   │   ├── splitting.py             # Subject-exclusive splitting and LOSO folds
│   │   └── preprocess_opportunity.py# CLI end-to-end dataset preprocessor
│   │
│   ├── features/
│   │   └── body_regions.py          # Anatomical grouping and skeletal graph prior
│   │
│   ├── models/
│   │   ├── classical.py             # Statistical feature extraction + RF / SVM / LR
│   │   ├── cnn.py                   # 1D CNN baseline
│   │   ├── fixed_gnn.py             # Fixed Anatomical GCN
│   │   └── adaptive_gnn.py          # Activity-Adaptive Anatomical GNN
│   │
│   ├── training/                    # GPU training engines with strict CUDA guards
│   │   ├── train_cnn.py
│   │   ├── train_fixed_gnn.py
│   │   └── train_adaptive_gnn.py
│   │
│   ├── evaluation/
│   │   └── metrics.py               # Accuracy, Macro F1, Weighted F1, Confusion Matrix
│   │
│   └── experiments/                 # Automated experimental runners
│       ├── run_baseline.py          # Classical baseline runner
│       ├── run_ablation.py          # Anatomical ablation experiment
│       ├── run_robustness.py        # Sensor failure and outage study
│       └── run_loso.py              # Leave-One-Subject-Out evaluation
│
├── tests/                           # 29 automated pytest unit tests
├── check_environment.py             # Hardware environment checker
├── requirements.txt
└── README.md
```

---

## 📈 Empirical Results Summary

### 1. Classical Machine Learning Baselines (CPU)
Evaluated on Locomotion recognition using 532 statistical features extracted from 133 on-body channels:

| Model | Accuracy | Macro F1 | Weighted F1 | Macro Precision | Macro Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **67.66%** | **54.62%** | **63.74%** | **60.06%** | **55.57%** |
| **Logistic Regression** | 51.95% | 45.49% | 52.37% | 60.07% | 45.74% |
| **Linear SVM** | 43.47% | 45.85% | 47.02% | 58.69% | 42.32% |

### 2. Anatomical Body-Region Ablation Study (CPU)
Assessing the impact of isolating or removing individual anatomical regions:

| Ablation Scenario | Channels Kept | Accuracy | Macro F1 | Weighted F1 |
| :--- | :---: | :---: | :---: | :---: |
| **All Sensors (Clean)** | **133** | **67.66%** | **54.62%** | **63.74%** |
| **Without Trunk** | 114 | 67.78% | 54.99% | 64.20% |
| **Without Right Arm** | 95 | 66.64% | 53.63% | 62.75% |
| **Without Left Arm** | 95 | 66.78% | 53.52% | 63.16% |
| **Without Right Leg** | 111 | 67.94% | 54.72% | 64.62% |
| **Without Left Leg** | 117 | 66.25% | 53.03% | 62.03% |

### 3. Sensor-Loss Robustness Study (CPU)
Degradation evaluated under random sensor dropout and complete regional failure:

| Failure Scenario | Dropped Channels | Accuracy | Macro F1 | Degradation |
| :--- | :---: | :---: | :---: | :---: |
| **0% Failure (Clean)** | 0 | 67.66% | 54.62% | 0.00% |
| **10% Random Dropout** | 13 | 66.58% | 53.69% | -0.92% |
| **25% Random Dropout** | 33 | 60.63% | 46.56% | -8.05% |
| **50% Random Dropout** | 66 | 53.95% | 37.21% | -17.41% |
| **75% Random Dropout** | 99 | 35.80% | 10.54% | -44.07% |
| **Complete Right Leg Outage** | 22 | 41.90% | 20.77% | **-33.85%** |
| **Complete Left Arm Outage** | 38 | 60.07% | 46.03% | -8.59% |

### 4. Leave-One-Subject-Out (LOSO) Cross-Validation (CPU)
Evaluating cross-subject generalization across 4 folds (Random Forest):

| Held-Out Subject | Accuracy | Macro F1 | Weighted F1 | Macro Precision | Macro Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **S1** | 77.96% | 80.46% | 77.55% | 80.21% | 82.60% |
| **S2** | 76.21% | 75.89% | 76.15% | 81.78% | 72.56% |
| **S3** | 69.70% | 53.62% | 66.57% | 57.51% | 53.68% |
| **S4** | 79.09% | 79.31% | 78.76% | 85.22% | 78.46% |
| **Mean** | **75.74%** | **72.32%** | **74.76%** | **76.18%** | **71.82%** |
| **Std Dev** | ±4.19% | ±12.62% | ±5.56% | ±12.62% | ±12.78% |

---

## 💻 Reproduction Instructions

### 1. Environment Setup
```bash
git clone <repo-url>
cd Opportunity_HAR
pip install -r requirements.txt
python check_environment.py
```

### 2. Preprocessing & Dataset Verification (CPU)
```bash
# Inspect dataset and generate metadata JSONs
python -m src.data.inspect_opportunity

# Run leak-free preprocessing and sliding window segmentation
python -m src.data.preprocess_opportunity --track Locomotion
```

### 3. CPU Baseline & Analytical Experiments
```bash
# Run classical baselines (RF, LR, SVM)
python -m src.experiments.run_baseline

# Run anatomical body-region ablation
python -m src.experiments.run_ablation

# Run sensor failure and robustness study
python -m src.experiments.run_robustness

# Run 4-subject Leave-One-Subject-Out (LOSO) cross-validation
python -m src.experiments.run_loso --model random_forest
```

### 4. Deep Learning & Graph Models (Colab GPU)
Open `notebooks/02_gpu_training.ipynb` in Google Colab or execute on a CUDA-equipped machine:
```bash
# Train 1D CNN Baseline
python -m src.training.train_cnn --config configs/cnn.yaml

# Train Fixed Anatomical GNN
python -m src.training.train_fixed_gnn --config configs/fixed_gnn.yaml

# Train Activity-Adaptive Anatomical GNN
python -m src.training.train_adaptive_gnn --config configs/adaptive_gnn.yaml
```

To run a fast 1-batch smoke test on CPU without training:
```bash
python -m src.training.train_cnn --smoke_test
python -m src.training.train_fixed_gnn --smoke_test
python -m src.training.train_adaptive_gnn --smoke_test
```

### 5. Automated Unit Tests
```bash
python -m pytest tests/ -v
```
All **29 unit tests pass**, verifying loaders, scalers, windowing, body mappings, forward passes, and metrics.
