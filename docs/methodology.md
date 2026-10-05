# Research Methodology: OPPORTUNITY Human Activity Recognition

This document details the scientific methodology, data integrity standards, and validation protocols used throughout this project.

---

## 1. Dataset Verification & Ingestion

- **Dataset**: UCI OPPORTUNITY Activity Recognition Benchmark.
- **Subjects**: 4 human subjects ($S1, S2, S3, S4$).
- **Runs per Subject**: 5 Natural Activities of Daily Living ($ADL1 - ADL5$) and 1 predefined scripted $Drill$ run per subject (24 total recording runs).
- **Sampling Frequency**: Strictly verified at **30 Hz** (nominal timestamp delta of ~33.33 ms).
- **Sensory Modalities**: 133 on-body wearable channels consisting of:
  - 12 Custom Triaxial Accelerometers (Knees, Hip, Upper Arms, Lower Arms, Wrists, Hands, Back).
  - 7 Commercial Inertial Measurement Units (accel, gyro, mag, quaternions on Back, Limbs, and Shoes).
- **Target Task**: Locomotion recognition (Null, Stand, Walk, Sit, Lie) and Gestures (17 mid-level manipulative actions).

---

## 2. Leak-Free Preprocessing & Scaling

1. **Missing Value Imputation**: Continuous sensor recordings contain short packet dropouts. Missing readings are linearly interpolated along continuous runs, with boundary NaNs filled with 0.0.
2. **Train-Only Normalization**:
   - `SensorStandardScaler` computes the feature mean ($\mu$) and standard deviation ($\sigma$) **strictly using training subject runs**.
   - These exact statistics are frozen and applied to transform validation and test subject recordings.
   - **Zero Data Leakage Rule**: Under no circumstances are test or validation statistics used during normalization.

---

## 3. Sliding-Window Segmentation

- Continuous recordings are segmented into fixed temporal windows of length $T = 30$ samples (1.0 second duration) with a stride $S = 15$ samples (50% overlap).
- **Boundary Preservation**: Windowing is performed *per continuous run*. Windows are never allowed to cross boundaries between different runs or subjects.
- **Label Assignment**: The activity label for each window is assigned using the `mode` (majority vote) across the 30 temporal steps.

---

## 4. Evaluation Protocols

1. **Primary Metric**: **Macro F1-Score** $\left(\frac{1}{K} \sum_{k=1}^K F1_k\right)$.
   - Due to the natural class imbalance in opportunistic daily living activities (e.g. standing/walking vs lying down), Macro F1 is the definitive research standard because it weights all activity classes equally.
2. **Secondary Metrics**: Sample-wise Accuracy, Weighted F1, Macro Precision, Macro Recall, and full Confusion Matrices.
3. **Validation Strategy**:
   - Standard Benchmark Split: Train on $S1$ (all runs), $S2$ (ADL1-3, Drill), $S3$ (ADL1-3, Drill); Validate on $S2$ (ADL4-5); Test on $S3$ (ADL4-5).
   - Leave-One-Subject-Out (LOSO): Iteratively train on 3 subjects and evaluate on the 4th held-out subject to evaluate cross-subject generalization.
