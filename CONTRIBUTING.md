# Contributing to Opportunity_HAR

Thank you for your interest in contributing to the **Opportunity_HAR** research project! We welcome contributions to models, preprocessing optimizations, evaluation protocols, and documentation.

---

## 🛠️ Development Guidelines

1. **Research Integrity First**:
   - Never fabricate or guess sensor metadata or dataset assumptions.
   - Ground all dataset mappings strictly in official UCI OPPORTUNITY documentation (`column_names.txt`, `label_legend.txt`).
   - Maintain strict leak-free normalization (scalers must fit on training data only).
2. **Compute Safety**:
   - Keep laptop CPU runs safe: neural network full training requires CUDA GPU.
   - Provide `--smoke_test` or `EXPERIMENT_MODE = "DEBUG"` for 1-batch CPU verification tests.
3. **Code Style & Testing**:
   - Write clear, type-annotated, modular Python code.
   - Ensure all tests pass prior to submitting a pull request:
     ```bash
     python -m pytest tests/ -v
     ```
   - Add unit tests for any newly added features, loaders, or model architectures in `tests/`.

---

## 🚀 Pull Request Process

1. Fork the repository and create your feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Commit your changes with descriptive messages:
   ```bash
   git commit -m "Add: feature description"
   ```
3. Push to your branch and open a Pull Request against `main`.
4. Ensure the GitHub Actions CI tests pass.
