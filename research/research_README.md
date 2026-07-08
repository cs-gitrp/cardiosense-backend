# CardioSense AI — Research Pipeline

This directory contains the complete research pipeline used to develop the
production CardioSense AI system.

Unlike the production backend (`app/`), this folder preserves the experimental
notebooks, evaluation outputs, and model development process that led to the
final deployed pipeline. It is organized to be independently reproducible.

---

## Key Results

| Metric                         | Value             |
|-------------------------------|-------------------|
| Clinical branch AUC            | 0.9266 (95% CI: 0.8887–0.9582) |
| ECG branch AUC                 | 0.9424 (95% CI: 0.9347–0.9504) |
| Fused pipeline AUC             | 0.9582 (95% CI: 0.9445–0.9719) |
| Clinical Brier score           | 0.1130            |
| ECG Brier score                | 0.0943            |
| Clinical ECE (calibration)     | 0.0653            |
| ECG ECE (calibration)          | 0.0340            |
| Recall (clinical branch)       | 90.2% (catches 92/102 positive cases) |
| Multiclass severity F1 (macro) | 0.3292 (known limitation — small per-class N) |

Bootstrap CIs computed over 1,000 resamples. Full metric tables in `results/metrics/`.

---

## Research Workflow

```
Notebook 01   Data cleaning and clinical preprocessing
     ↓
Notebook 02   Patient-level train/validation/test split (GroupShuffleSplit)
     ↓        → 02b: Corrected KNN imputation (fixed data leakage from initial run)
Notebook 03   Baseline clinical machine learning models
     ↓
Notebook 04   Consensus feature selection (Chi-Square + LASSO + RF, top-10-in-2-of-3)
     ↓        → Frozen feature set: 11 features → results/metrics/04_selected_features.csv
Notebook 05   Feature ablation study (17-feature baseline vs 11-feature selection)
     ↓
Notebook 06   PTB-XL ECG dataset acquisition and exploration
     ↓
Notebook 07   ECG signal preprocessing (100 Hz, 10s, 12-lead normalization)
     ↓
Notebook 08   CNN ECG model training — ablation: CNN-only vs CNN-LSTM
     ↓        → CNN-only wins (AUC 0.9162 vs 0.8893); architecture locked
     ↓        → 08b: Full PTB-XL training run (21,437 records)
Notebook 09   Probability calibration (Platt scaling) + confidence-adaptive fusion
     ↓        → gamma=3 selected via sensitivity sweep (γ=1,2,3,5)
Notebook 10   Explainability — SHAP (clinical RF) + Integrated Gradients (ECG CNN)
     ↓
Notebook 11   Bootstrap confidence intervals (n=1,000, threshold=0.50)
     ↓
Notebook 12   Production inference pipeline assembly
     ↓        → Output: ml_artifacts/cardiosense_pipeline/ (used by FastAPI backend)
Notebook 13   Severity grading extension (multiclass 0–4)
              → Documented limitation: Grade 4 F1=0.00 (n=6 test samples)
```

---

## PROCESS.md

`PROCESS.md` is a running log of architectural decisions, bugs caught and fixed,
and pivots made during development. It is the authoritative record of *why* things
were built the way they were, not just *what* was built.

Key decisions logged:
- KNN imputation leakage discovery and fix (Notebook 02b)
- CNN-only vs CNN-LSTM ablation result
- Gamma sensitivity sweep (fusion module)
- `chol=0` and `trestbps=0` treated as missing (not biological zeros)
- Feature selection rule committed before seeing results (pre-registered decision)

---

## Folder Overview

```
research/
  README.md           This file
  PROCESS.md          Decision log and development journal

  notebooks/          Jupyter notebooks (01–13) in execution order

  data/
    README.md         Dataset download instructions (data not included)

  results/
    metrics/          CSV files: per-notebook evaluation outputs, feature rankings
    plots/            Publication-quality figures (PNG/PDF)
    tables/           LaTeX / markdown tables used in documentation
    manifest/         Experiment metadata and configuration records
    trained_models/   Research checkpoints (NOT used in production)
    arrays/           Large numerical artifacts that cannot be cheaply recomputed
      10_integrated_gradients.npy
```

---

## Production Components

The deployed application does **not** use this directory.

Production inference loads only:

```
ml_artifacts/cardiosense_pipeline/
  cardiosense_pipeline.pkl
  clinical_rf.pkl
  ecg_model.keras
  09_confidence_adaptive_fusion.pkl
  04_selected_features.csv
  manifest.json
  pipeline_class.py
```

These artifacts were produced by Notebook 12 and are loaded by the FastAPI backend
(`app/services/model_loader.py`). The research notebooks and production pipeline
are intentionally decoupled — notebooks can be rerun without affecting the
deployed service.

---

## Datasets

The original datasets are intentionally excluded due to size and licensing.

See `research/data/README.md` for download instructions and preprocessing notes.

---

## Computational Requirements

| Component              | Environment used        | Approx. runtime |
|------------------------|-------------------------|-----------------|
| Notebooks 01–05        | Google Colab (CPU)      | < 10 min total  |
| Notebook 08 (5K dev)   | Kaggle (T4 GPU)         | ~20 min         |
| Notebook 08b (full)    | Kaggle (T4 GPU)         | ~2–3 hours      |
| Notebooks 09–13        | Kaggle / Colab (CPU)    | < 30 min total  |

Notebooks 01–07 and 09–13 run on CPU. Only Notebook 08/08b requires GPU.

---

## Reproducibility

Every result in the accompanying paper can be reproduced from these notebooks
using the publicly available datasets and the preprocessing pipeline documented
in Notebooks 01–02.

Fixed random seeds: `SEED = 42` throughout.  
Patient-grouped splits prevent any patient identity from appearing in both
training and test sets (`GroupShuffleSplit` on patient ID column).
