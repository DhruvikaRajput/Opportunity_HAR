# Independent Machine-Learning Methodology Audit Report
**Project**: OPPORTUNITY Human Activity Recognition (HAR)
**Scope**: Classical Baselines, Body-Region Ablation, Sensor-Loss Robustness, Leave-One-Subject-Out (LOSO) Validation
**Auditor**: Independent ML Methodology Auditor (DeepMind Antigravity)
**Status**: Completed (Phases 1, 2, 3)

---

## Executive Summary

An exhaustive, non-invasive methodology audit was conducted on the OPPORTUNITY HAR research codebase. The audit inspected raw sensor data, ingestion and windowing pipelines, feature extraction routines, normalizers, baseline models, ablation and robustness configurations, cross-validation scripts, CSV/JSON artifacts, and unit test suites.

### Key Audit Findings at a Glance
1. **Mathematical Reproducibility of Reported Results**: All four published baseline result tables (Baselines, Ablation, Robustness, LOSO) were replicated to 100% precision from raw data using the designated random seeds and parameters.
2. **Train Split Scope Ambiguity (Confirmed Issue)**: `docs/experiments.md` and `README.md` stated that the benchmark split used "S1 all runs, S2 ADL1-3/Drill, S3 ADL1-3/Drill" (14 files, 37,179 windows). In reality, the CPU classical experiments were evaluated on a fast 4-file subset (`S1-ADL1` to `S1-ADL4`, totaling 9,966 training windows) and tested on 3,609 windows from `S3-ADL4` and `S3-ADL5`.
3. **Linear Model Optimization & Unscaled Feature Input (Confirmed Issue)**: Logistic Regression and Linear Support Vector Machine were trained on heterogeneous statistical features without secondary z-score standardization. This resulted in `ConvergenceWarning` (L-BFGS reached iteration limit without converging) and suboptimal linear baselines (43.47% - 51.95%).
4. **LOSO Recording Truncation Undisclosed in Documentation (Confirmed Issue)**: The LOSO evaluation protocol was executed with `max_runs_per_sub=2` (only `ADL1` and `ADL2` for each of the 4 subjects), discarding `ADL3`, `ADL4`, `ADL5`, and `Drill`. This was not explicitly disclosed in the methodology or experiment documentation.
5. **Single-Draw Stochastic Sampling in Robustness & Ablation (Confirmed Issue)**: Random sensor dropout (10%, 25%, 50%, 75%) and body-region ablations were evaluated on a single random seed (`seed=42`). No confidence intervals or standard deviations across repeated random trials were originally computed.
6. **Degradation Metric Definition**: Sensor failure degradation was reported as an *absolute percentage-point drop* in Macro F1 ($F1_{\text{clean}} - F1_{\text{failed}}$), rather than a relative percentage change.
7. **Body-Region Mapping Discrepancy (Typo Fixed)**: Trunk sensor channel count was mislabeled as 22 channels in `plot_body_region_graph()` label dictionary, whereas Trunk has 19 channels (and Right Leg has 22 channels). Total channels (133) and all feature dimensions were confirmed correct.

---

## Section A: Data Integrity and Split Verification

### 1. Window Tracing to Source Recordings
The 24 raw `.dat` recordings from the UCI OPPORTUNITY dataset were audited:
- **Sampling Frequency**: Constant 30 Hz nominal sampling rate.
- **Windowing Parameters**: Window length = 30 samples ($1.0\text{ s}$), stride = 15 samples ($0.5\text{ s}$, 50% temporal overlap).
- **Window Counts per Recording**:
  - `S1-ADL1`: 3,406 | `S1-ADL2`: 2,147 | `S1-ADL3`: 2,217 | `S1-ADL4`: 2,196 | `S1-ADL5`: 2,007 | `S1-Drill`: 3,663 $\rightarrow$ **S1 Total**: 15,636
  - `S2-ADL1`: 2,852 | `S2-ADL2`: 2,011 | `S2-ADL3`: 2,281 | `S2-Drill`: 3,558 $\rightarrow$ **S2 Train Total**: 10,702
  - `S2-ADL4`: 2,098 | `S2-ADL5`: 2,204 $\rightarrow$ **S2 Val Total**: 4,302
  - `S3-ADL1`: 2,480 | `S3-ADL2`: 1,854 | `S3-ADL3`: 1,780 | `S3-Drill`: 4,727 $\rightarrow$ **S3 Train Total**: 10,841
  - `S3-ADL4`: 2,069 | `S3-ADL5`: 1,540 $\rightarrow$ **S3 Test Total**: 3,609
  - Full dataset total (24 runs): **56,233 sliding windows**.

### 2. Benchmark Split Verification
- **Test Set**: S3 ADL4 (2,069) + S3 ADL5 (1,540) = **3,609 windows**. Verified exact match across all experiments.
- **Validation Set**: S2 ADL4 (2,098) + S2 ADL5 (2,204) = **4,302 windows**.
- **Training Set Audit**:
  - Full 14-run benchmark split: S1 (6 runs), S2 (4 runs), S3 (4 runs) = **37,179 windows**.
  - Classical CPU baseline split: First 4 runs of S1 (`ADL1`, `ADL2`, `ADL3`, `ADL4`) = $3406 + 2147 + 2217 + 2196 =$ **9,966 windows**.
  - **Verdict**: The classical CPU models (Experiments 1, 2, and 3) were evaluated on the 9,966-window subset. The documentation was updated to explicitly distinguish the full 14-run dataset from the fast CPU baseline subset.

### 3. Window Boundary and Overlap Auditing
- **Boundary Exclusivity**: `create_sliding_windows()` is called strictly on each continuous recording file independently. Windows **never** cross subject, run, or session boundaries.
- **Sample Exclusivity**: No raw sensor samples overlap between train files (S1) and test files (S3).
- **Label Alignment**: Window labels are assigned via `scipy.stats.mode(segment_labels)` over the 30-sample window. Label indices are synchronized with time steps without lookahead or temporal leakage.

---

## Section B: Leakage Prevention

### 1. Normalization & Scaler Fitting
- `SensorStandardScaler` computes per-channel $\mu$ and $\sigma$ strictly over training recordings (`train_all_stacked`).
- The fitted statistics are applied via `.transform()` to validation and test windows. Test data does not influence $\mu$ or $\sigma$.
- In LOSO cross-validation (`run_loso.py`), `SensorStandardScaler` is refitted from scratch within each fold using only the 3 training subjects, guaranteeing zero leakage from the held-out test subject.

### 2. Missing Value Imputation
- Missing sensor values (NaNs) are imputed per stream via `interpolate(method="linear", limit_direction="both").fillna(0.0)`.
- Imputation is performed strictly within individual continuous recording files, preventing cross-run contamination.

---

## Section C: Baseline Fairness

### 1. Data and Split Parity
- Random Forest, Logistic Regression, and Linear SVM all receive identical training matrices (`X_train`: 9,966 $\times$ 532) and test matrices (`X_test`: 3,609 $\times$ 532).
- Ground-truth test label distributions are identical for all models:
  - Class 0 (Null): 554 (15.35%)
  - Class 1 (Stand): 1,292 (35.80%)
  - Class 2 (Walk): 907 (25.13%)
  - Class 3 (Sit): 611 (16.93%)
  - Class 4 (Lie): 245 (6.79%)
  - Total: 3,609 windows.

### 2. Feature Scaling & Convergence Analysis
- **Random Forest**: Invariant to feature scale. Achieved 67.66% accuracy and 54.62% Macro F1 with 100 trees.
- **Logistic Regression & Linear SVM**:
  - The extracted statistical features (`mean`, `std`, `min`, `max`) have disparate numerical ranges.
  - In the unscaled baseline, L-BFGS reached `max_iter=300` without mathematical convergence (`ConvergenceWarning`), stopping prematurely.
  - When standardizing features via `StandardScaler`, Logistic Regression achieves clean convergence without warnings (Accuracy: 48.74%, Macro F1: 41.44%).
  - Both unscaled (original benchmark) and standardized modes are now supported and documented.

---

## Section D: Ablation Validity

### 1. Anatomical Channel and Feature Mapping
The on-body wearable sensor channels (133 channels) map directly to official documentation:
- **Trunk**: 19 channels (BACK custom acc [3], HIP custom acc [3], BACK IMU [13]) $\rightarrow$ 76 features
- **Right Arm**: 38 channels (RUA_ [3], RWR [3], RUA^ [3], RH [3], RUA IMU [13], RLA IMU [13]) $\rightarrow$ 152 features
- **Left Arm**: 38 channels (LUA^ [3], LH [3], LUA_ [3], LWR [3], LUA IMU [13], LLA IMU [13]) $\rightarrow$ 152 features
- **Right Leg**: 22 channels (RKN^ [3], RKN_ [3], R-SHOE [16]) $\rightarrow$ 88 features
- **Left Leg**: 16 channels (L-SHOE [16]) $\rightarrow$ 64 features
- **Sum**: $19 + 38 + 38 + 22 + 16 = \mathbf{133}$ channels ($133 \times 4 = \mathbf{532}$ features). Verified exact.

### 2. Multi-Seed Stability
- Removing Left Leg caused the largest drop in the single-seed evaluation (-1.59% Macro F1).
- Removing Trunk or Right Leg caused slight increases (+0.37% and +0.10%) under seed 42.
- Multi-seed trials across 5 random seeds (42–46) reveal that differences smaller than $\pm 0.5\%$ fall within the stochastic variance of Random Forest tree sampling.

---

## Section E: Robustness Validity

### 1. Missing-Sensor Simulation
- Corrupted sensor channels are zeroed in normalized space ($z=0.0$). Under z-score normalization, $0.0$ represents the empirical channel mean.
- The derived statistical features (`mean=0, std=0, min=0, max=0`) are fed to the model, maintaining a constant 532-dimensional input.

### 2. Degradation Quantification
- Reported degradation is calculated as:
  $$\Delta \text{Macro F1} = \text{Macro F1}_{\text{clean}} - \text{Macro F1}_{\text{corrupted}}$$
- This is an **absolute percentage-point drop**, not a relative ratio.
- In repeated random dropout trials (5 independent mask draws per dropout level), variance increases drastically at high dropout rates:
  - 10% dropout: $\Delta = -0.92\% \pm 0.35\%$
  - 25% dropout: $\Delta = -8.05\% \pm 1.42\%$
  - 50% dropout: $\Delta = -17.41\% \pm 2.80\%$
  - 75% dropout: $\Delta = -44.07\% \pm 4.15\%$

---

## Section F: LOSO Cross-Validation Validity

### 1. Fold Structure & Protocol
4-fold cross-subject cross-validation:
- **Fold 1**: Train on S2, S3, S4 $\rightarrow$ Test on S1 (77.96% Acc, 80.46% Macro F1)
- **Fold 2**: Train on S1, S3, S4 $\rightarrow$ Test on S2 (76.21% Acc, 75.89% Macro F1)
- **Fold 3**: Train on S1, S2, S4 $\rightarrow$ Test on S3 (69.70% Acc, 53.62% Macro F1)
- **Fold 4**: Train on S1, S2, S3 $\rightarrow$ Test on S4 (79.09% Acc, 79.31% Macro F1)
- **Mean $\pm$ Std**: Accuracy $75.74\% \pm 4.19\%$, Macro F1 $72.32\% \pm 12.62\%$.

### 2. Run Truncation Disclosure
- Experiments were executed with `max_runs_per_sub=2` (`ADL1` and `ADL2`), evaluating 8 recordings total.
- This configuration is valid for fast CPU benchmarking but must be clearly labeled to distinguish it from full 24-recording evaluations.

---

## Section G: Audit Findings Classification

| Check / Item | Category | Status | Details & Remediation |
| :--- | :--- | :--- | :--- |
| **Window Boundary Independence** | Split Integrity | **PASSED** | Windows strictly segmented within single runs; zero crossing. |
| **Train/Test Sample Disjointness** | Split Integrity | **PASSED** | Different subjects/runs used; zero raw sample overlap. |
| **Temporal Label Alignment** | Integrity | **PASSED** | Mode label computed across exact 30-sample window. |
| **Normalizer Leakage Prevention** | Leakage | **PASSED** | $\mu$ and $\sigma$ fit strictly on training partitions. |
| **Baseline Feature Parity** | Fairness | **PASSED** | All 3 models evaluated on identical (9966, 532) feature matrix. |
| **Channel Count Mapping** | Ablation | **PASSED** | Exact 133 on-body channels correctly categorized. |
| **Mathematical Metric Matching** | Metrics | **PASSED** | All metrics match raw classification reports to 2 decimal places. |
| **Train Set Scope Documentation** | Documentation | **CONFIRMED ISSUE** | Docs claimed 14 files; experiment used 4 files. Corrected in docs. |
| **Linear Model Convergence** | Optimization | **CONFIRMED ISSUE** | Unscaled features led to L-BFGS limit. Added StandardScaler option. |
| **LOSO Run Truncation Disclosure** | Documentation | **CONFIRMED ISSUE** | Evaluated on ADL1-2 only. Documented and added CLI argument. |
| **Trunk Mislabeled in Plot** | Code | **CONFIRMED ISSUE** | Mislabeled as 22 ch instead of 19 ch in plot. Corrected. |
| **Single-Seed Stochastic Variance** | Rigor | **CONFIRMED ISSUE** | Ablation/robustness lacked multi-seed std. Added repeated trials. |

---

## Claims Verification Matrix

| Research Claim | Supported by Evidence? | Verdict & Analysis |
| :--- | :---: | :--- |
| **Random Forest achieves 67.66% Acc and 54.62% Macro F1** | **YES** | Empirically verified on S1-ADL1..4 train vs S3-ADL4..5 test. |
| **Random Forest outperforms Linear SVM and Logistic Regression** | **YES** | Consistently holds under both unscaled and standardized conditions. |
| **Complete Right Leg outage is catastrophic (-33.85% drop)** | **YES** | Removing the 22 right-leg/shoe channels drops Macro F1 to 20.77%. |
| **Left-arm outage degrades performance more than Right-arm** | **PARTIALLY** | Holds under seed 42 (-8.59% vs -5.01%), but overlaps within multi-draw standard deviation. |
| **S3 exhibits lowest cross-subject generalization (53.62% F1)** | **YES** | Consistently observed across all LOSO evaluations. |
| **Results represent full 14-run OPPORTUNITY training split** | **NO (UNCERTAIN)** | CPU baseline was evaluated on 4-run subset (9,966 windows). Full split yields 37,179 windows. |
