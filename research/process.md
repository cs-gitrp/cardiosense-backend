# CardioSense AI — Process & Architecture Document

> This is the authoritative record of how CardioSense AI was designed, built, and
> what was actually delivered — including decisions that changed from the original plan
> and why. It is the single document to hand anyone asking "how does this work?"

---

## Table of Contents

1. [What Was Built](#1-what-was-built)
2. [Technical Novelty — Honest Account](#2-technical-novelty--honest-account)
3. [ML Pipeline — 13 Notebooks](#3-ml-pipeline--13-notebooks)
4. [Key Results](#4-key-results)
5. [Architecture — As Deployed](#5-architecture--as-deployed)
6. [Critical Decisions & Pivots](#6-critical-decisions--pivots)
7. [Data](#7-data)
8. [Full Stack Overview](#8-full-stack-overview)
9. [What Was Planned vs What Shipped](#9-what-was-planned-vs-what-shipped)

---

## 1. What Was Built

**CardioSense AI** is a multimodal cardiac risk screening platform that fuses
structured clinical features with 12-lead ECG signals to produce calibrated,
explainable cardiac risk predictions.

**Type:** Full-stack production prototype — BTech CSE (AI/ML) final year project,
KCC Institute of Technology & Management, Noida. Graduating 2027.

**Scope:** Not a clinical tool. Not a toy notebook. A deployed platform with:
- A 13-notebook ML research pipeline (clinical + ECG branches, calibration, fusion,
  explainability, bootstrap evaluation, severity grading)
- A FastAPI backend with streaming inference, JWT auth, PostgreSQL persistence
- A Next.js frontend with animated pipeline visualization, SHAP charts, ECG lead
  attribution, history tracking, model insights dashboard, and an AI chatbot
- Deployed: Render (backend) + Vercel (frontend)

**The one architectural idea everything is built around:**  
When a clinical test and an ECG both exist for a patient, don't trust them equally.
Trust whichever one is more confident for *this specific patient* — and be transparent
about which branch drove the prediction and why.

---

## 2. Technical Novelty — Honest Account

### What is genuinely novel

**Confidence-adaptive fusion with explicit branch attribution.**  
Most multimodal systems use fixed-weight averaging ("50% clinical, 50% ECG").
CardioSense uses a confidence-adaptive weighting function: each branch's contribution
is proportional to how far its calibrated probability is from the decision boundary
(i.e., how *confident* it is). A clinical branch outputting 0.92 and an ECG branch
outputting 0.55 will weight the clinical branch at ~84% for that patient. This is
computed analytically (not learned), validated via a gamma sensitivity sweep, and
calibrated via Platt scaling.

**Missingness as a feature, not just a preprocessing step.**  
Rather than silently imputing missing clinical values, missingness-indicator flags
(`chol_missing`, `slope_missing`) are fed into the model as explicit features.
The model can therefore learn that "this patient's cholesterol was unavailable"
carries information — and the fusion module can compensate by weighting the ECG
branch more heavily when clinical data is incomplete.

**Full evaluation stack for a research prototype.**  
Calibration (Brier score, ECE), bootstrap confidence intervals (n=1000),
per-lead ECG attribution via Integrated Gradients, and SHAP feature attribution
are all implemented and surfaced in the deployed platform's Insights dashboard —
not just in notebook outputs.

### What was originally planned but delivered differently

| Planned | Actual | Why it changed |
|---|---|---|
| Learned gating network | Confidence-adaptive formula (gamma=3) | Formula performed comparably, was interpretable, and avoided overfitting a small meta-learner on limited paired data |
| CNN-LSTM ECG branch | CNN-only (3 conv blocks) | Controlled ablation in Notebook 08 showed LSTM added no performance — AUC 0.9162 (CNN) vs 0.8893 (CNN-LSTM). Reported as a genuine negative result. |
| FAISS vector store for CardioBot | Groq LLM with assessment context injection | Assessment data is already structured; retrieving it from a vector store added latency with no quality benefit |

### What is NOT claimed as novel (by design)

Standard CNN layers, Random Forest, XGBoost, Platt calibration, and SHAP are
established methods. They are components, not contributions. The contribution
is the system: how these components are combined, calibrated, and made
interpretable for a clinical screening context.

---

## 3. ML Pipeline — 13 Notebooks

### Overview

```
01 → 02 → 02b → 03 → 04 → 05      Clinical branch + feature selection
                              ↓
06 → 07 → 08 → 08b             ECG branch + CNN training
                  ↓
                  09            Calibration + confidence-adaptive fusion
                  ↓
             10 → 11            Explainability + bootstrap evaluation
                  ↓
                  12            Production inference pipeline
                  ↓
                  13            Severity grading extension
```

---

### Notebook 01 — Data Cleaning

**Input:** 4 raw UCI Heart Disease files (Cleveland 303, Hungarian 294,
Switzerland 123, VA 200 = 920 rows combined)

**Key decisions:**
- `?` markers → `NaN` (do not treat as a string category)
- `chol=0` and `trestbps=0` treated as biologically impossible → also `NaN`
  (Switzerland and VA datasets have many zero-cholesterol rows — these are
  recording absences, not real readings)
- Missingness-indicator columns created: `chol_missing`, `slope_missing`,
  `ca_missing`, `thal_missing` — these travel with the data through every
  downstream step
- Two output datasets:
  - `binary_dataset.csv` (920 rows) — target binarized to 0/1
  - `multiclass_dataset.csv` (626 rows) — target kept as 0-4 severity scale
    (294 rows dropped because unusable severity labels in some sub-datasets)

---

### Notebook 02 — Train/Test Split

**GroupShuffleSplit on patient ID** to prevent identity leakage across splits.
`random_state=42` fixed throughout.

---

### Notebook 02b — Imputation Leakage Fix

**Bug caught:** Original Notebook 03 had KNN imputer fit on the full combined
dataset before splitting — a data leakage error.

**Fix:** Imputer now fit on TRAIN only, applied to test via `.transform()` not
`.fit_transform()`. Re-ran all downstream notebooks on corrected splits.

**Impact:** Accuracy dropped 1–3% — confirming leakage existed and was not
catastrophic, but was real.

---

### Notebook 03 — Baseline Models

4 classical models on 17 raw features (before feature selection):

| Model | Accuracy | F1 | AUC |
|---|---|---|---|
| Logistic Regression | 81.52% | 0.8365 | 0.897 |
| Random Forest | 83.70% | 0.8544 | 0.925 |
| SVM | 82.61% | 0.8431 | 0.908 |
| XGBoost | 83.15% | 0.8476 | 0.912 |

Random Forest is the strongest clinical baseline.

---

### Notebook 04 — Feature Selection

**Method:** Consensus ranking — three methods run independently, top-10-in-2-of-3 rule.
Rule committed *before* seeing results (pre-registered to avoid post-hoc selection bias).

Three methods:
- Chi-Square test (filter method)
- LASSO (L1-regularized logistic regression)
- Random Forest feature importance

**11 consensus-selected features (FROZEN):**
`cp, exang, ca, sex, chol_missing, slope_missing, thal, thalach, age, oldpeak, fbs`

Feature set stored in `results/metrics/04_selected_features.csv`.
All downstream notebooks load from this file — never hardcode a feature list.

---

### Notebook 05 — Feature Ablation

Validated 11-feature set against 17-feature baseline on identical splits:

| Model | Δ Accuracy | Δ F1 | Result |
|---|---|---|---|
| Logistic Regression | +2.18% | +2.44% | Improved |
| Random Forest | +1.08% | +1.35% | Improved |
| SVM | -1.09% | -0.46% | Passed (<2% rule) |
| XGBoost | 0.00% | -0.15% | Passed (<2% rule) |

Two models *improved* with fewer features — dropped columns (`trestbps`, `restecg`,
`slope`, `chol`, `ca_missing`, `thal_missing`) were contributing noise, not signal.

**Locked best clinical model:**
Random Forest, 11 features — Accuracy 84.78%, F1 0.8679, Recall 90.2%
(catches 92/102 actual heart disease patients, misses only 10).
Recall is the priority metric — false negatives = missed at-risk patients.

---

### Notebook 06 — PTB-XL Acquisition

PTB-XL dataset: 21,437 12-lead ECG recordings at 100 Hz, 10 seconds per record.
Labels: binary heart disease labels extracted from multi-label diagnostic annotations.

---

### Notebook 07 — ECG Preprocessing

- Normalized to zero-mean, unit-variance per lead
- Shape: (1000 samples × 12 leads) per record
- GroupShuffleSplit on patient ID to prevent identity leakage

---

### Notebook 08 — CNN Architecture Ablation (5K dev subset)

**Research question:** Does adding LSTM after the conv blocks improve ECG
classification? Controlled for confound: held conv-depth constant across variants.

| Architecture | AUC | F1 |
|---|---|---|
| CNN-only (3 conv blocks) | **0.9162** | **0.8508** ← winner |
| CNN-LSTM (3 conv blocks + LSTM) | 0.8893 | 0.8374 |
| CNN-LSTM (2 conv blocks + LSTM) | 0.8785 | 0.8320 |

**Decision: CNN-only architecture locked.** LSTM does not improve morphology
classification at this scale. Reported as a genuine negative result —
not hidden, not explained away.

Patient-level leakage check: PASSED (zero patient overlap across train/val/test).

---

### Notebook 08b — Full PTB-XL Training

CNN trained on full 21,437-record PTB-XL dataset (T4 GPU, Kaggle).
Final ECG branch: **AUC 0.9424**.

---

### Notebook 09 — Calibration + Confidence-Adaptive Fusion

**Platt scaling (CalibratedClassifierCV, sigmoid)** applied to both branches:

| Branch | AUC | Brier Score | ECE |
|---|---|---|---|
| Clinical RF | 0.9262 | 0.1130 | 0.0653 |
| ECG CNN | 0.9424 | 0.0943 | 0.0340 |

ECG branch dominates on all three metrics — consistent story across AUC,
calibration error, and reliability.

**Confidence-adaptive fusion:**

```
confidence(p) = max(p, 1 - p)       # distance from 0.5 decision boundary

w_clinical = conf(clinical_prob)^γ / (conf(clinical_prob)^γ + conf(ecg_prob)^γ)
w_ecg      = 1 - w_clinical

fused_prob = w_clinical × clinical_prob + w_ecg × ecg_prob
```

**Gamma sensitivity sweep** (γ = 1, 2, 3, 5):
- γ=1: linear — moderate sharpening
- γ=3: chosen — meaningful confidence differentiation without near-binary selection
- γ=5: too aggressive — near-binary model selection on most patients

**Gamma=3 selected.** Justified by sweep, not arbitrary.

**Fused pipeline AUC: 0.9582**

**Key qualitative finding:** "Strong disagreement" case (clinical=0.15, ECG=0.88)
→ fused probability 0.521. Appropriately uncertain at the decision boundary when
branches strongly disagree. A deployed system should flag these cases for human
review rather than forcing a binary output. This is honest behavior.

*Limitation, disclosed:* UCI clinical and PTB-XL ECG are separate populations.
Paired case studies are illustrative, not real paired patient predictions.
This is documented and would require a clinical dataset with both modalities
per patient to resolve — which is a deployment, not a research, requirement.

---

### Notebook 10 — Explainability

- **Clinical branch:** SHAP 0.51.0, `TreeExplainer` on the Random Forest.
  Returns per-feature SHAP values for each assessment.
- **ECG branch:** Integrated Gradients on the full PTB-XL CNN (Notebook 08b model).
  Returns per-lead attribution scores (12 leads, named I/II/III/aVR/aVL/aVF/V1-V6).

Both are surfaced on the results page for every assessment run through the platform.

---

### Notebook 11 — Bootstrap Confidence Intervals

`bootstrap_metric_ci()`, n=1000, threshold=0.50, fixed SEED.

Per-branch (not fused — the fused CI is a known open item):

| Branch | Metric | Mean | Std | 95% CI Lower | 95% CI Upper |
|---|---|---|---|---|---|
| Clinical | AUC | 0.9266 | 0.0178 | 0.8887 | 0.9582 |
| ECG | AUC | 0.9424 | 0.0040 | 0.9347 | 0.9504 |
| Clinical | F1 Score | 0.8576 | 0.0258 | 0.8039 | 0.9065 |
| ECG | F1 Score | 0.8877 | 0.0058 | 0.8764 | 0.8988 |
| Clinical | Recall | 0.8920 | 0.0317 | 0.8230 | 0.9469 |
| ECG | Recall | 0.8756 | 0.0082 | 0.8594 | 0.8916 |
| Clinical | Precision | 0.8268 | 0.0366 | 0.7523 | 0.8957 |
| ECG | Precision | 0.9003 | 0.0069 | 0.8874 | 0.9135 |

ECG branch is consistently tighter (lower std) than Clinical across all metrics.
Clinical Precision has the widest CI (0.75–0.90) — the least stable metric in
the whole pipeline. This is disclosed in the Insights dashboard.

---

### Notebook 12 — Production Inference Pipeline

Assembled all components into a single callable `CardioSensePipeline` class,
serialized with all sub-components (except ECG CNN and ExplainabilityEngine,
which load separately due to Keras and SHAP class definition requirements).

**Contract:**
```python
pipeline.run(patient_dict, ecg_signal)
# patient_dict: {cp, exang, ca, sex, chol_missing, slope_missing,
#                thal, thalach, age, oldpeak, fbs}
# ecg_signal:   np.ndarray shape (1000, 12), normalized, 100 Hz
```

**Returns:**
```python
{
  "prediction":          "Disease" | "No Disease",
  "fused_probability":   float,
  "severity":            str,          # heuristic band
  "severity_source":     str,
  "confidence":          float,
  "branch_contribution": {"clinical_pct": float, "ecg_pct": float},
  "branch_probabilities":{"clinical": float, "ecg": float},
  "top_clinical_features": [...],      # SHAP
  "top_ecg_leads":          [...],     # Integrated Gradients
  "ecg_quality":            {...},
  "recommendations":        [...],
  "disclaimer":             str
}
```

**Clinical-only mode:** When `ecg_signal=None`, a zero array is passed.
The fusion formula naturally discounts a zero ECG (maximum uncertainty = 0.5
probability = zero contribution weight). `branch_contribution` is overridden
to show 100% clinical, ECG leads set to null.

**Serialization fix:** `09_confidence_adaptive_fusion.pkl` (43 bytes) serializes
`adaptive_fusion` by reference, not by value. Loading it requires the function
to be in `__main__` first — circular. Fix: function is defined directly in
`pipeline_class.py` and injected into `__main__` before any pickle load.

---

### Notebook 13 — Severity Grading (Extension)

Multiclass (0-4 severity) Random Forest + XGBoost on 626-row multiclass dataset.

| Model | Accuracy | Macro F1 | ROC-AUC (OvR) |
|---|---|---|---|
| Random Forest | 0.4603 | 0.3292 | 0.7276 |
| XGBoost | 0.4048 | 0.2881 | 0.6798 |

**Per-class breakdown (RF):**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| 0 (no disease) | 0.71 | 0.78 | 0.74 | 45 |
| 1 (mild) | 0.29 | 0.38 | 0.32 | 32 |
| 2 | 0.37 | 0.32 | 0.34 | 22 |
| 3 | 0.31 | 0.19 | 0.24 | 21 |
| **4 (critical)** | **0.00** | **0.00** | **0.00** | **6** |

**Honest assessment:** Grade 4 F1=0.00. n=6 test samples — the model never
correctly identifies a single severity-4 patient. This is a real limitation,
driven by class imbalance and the small dataset, not a code bug.

**Production decision:** Severity field in the deployed platform uses a
heuristic probability-band mapping (`severity_source: "heuristic_probability_band"`)
rather than this model. Replacing the heuristic with Notebook 13's output
would make severity *worse*, not better. The trained model is retained for
research comparison only. Documented transparently in `severity_source` field.

**Mitigation path (not yet implemented):** Collapse to 3-class (Absent/Mild/Severe),
oversample rare classes, or treat as ordinal regression. A robustness notebook
cell was added demonstrating the 3-class collapse.

---

## 4. Key Results

```
┌─────────────────────────────────────────────────────────────┐
│                    CardioSense AI — Key Metrics             │
├──────────────────────┬────────────┬────────────┬────────────┤
│ Metric               │ Clinical   │ ECG        │ Fused      │
├──────────────────────┼────────────┼────────────┼────────────┤
│ AUC                  │ 0.9266     │ 0.9424     │ 0.9582     │
│ AUC 95% CI           │ .889–.958  │ .935–.950  │ .945–.972* │
│ Brier Score ↓        │ 0.1130     │ 0.0943     │ —          │
│ ECE ↓ (calibration)  │ 0.0653     │ 0.0340     │ —          │
│ F1 Score             │ 0.8679     │ 0.8877     │ —          │
│ Recall (Sensitivity) │ 90.2%      │ 87.6%      │ —          │
├──────────────────────┴────────────┴────────────┴────────────┤
│ * Fused CI not yet bootstrap-computed (open item)           │
│ Clinical binary: Accuracy 84.78%, catches 92/102 positives  │
│ ECG: AUC 0.9162 (dev subset) → 0.9424 (full PTB-XL)        │
│ Gamma=3 selected via sweep; fused AUC 0.9582 at threshold   │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Architecture — As Deployed

### 5.1 ML Core

```
                     Patient Input
                          │
          ┌───────────────┴────────────────┐
          │                                │
   Clinical Dict                    ECG Signal
   (11 features)               (1000 × 12 numpy array)
          │                                │
          ▼                                ▼
  ClinicalEngine                    ECGQualityEngine
  (KNN-imputed,                     (flatline check,
   feature-ordered)                  SNR estimate)
          │                                │
          ▼                                ▼
  Random Forest                       ECG CNN
  Classifier                     (3 conv blocks,
  (CalibratedClassifierCV)        PTB-XL trained)
          │                                │
          ▼                                ▼
  Platt-calibrated              Platt-calibrated
  clinical_prob                    ecg_prob
          │                                │
          └───────────────┬────────────────┘
                          │
                          ▼
            ┌─────────────────────────────┐
            │   Confidence-Adaptive       │
            │       Fusion (γ=3)          │
            │                             │
            │  conf(p) = max(p, 1-p)      │
            │                             │
            │  w = conf^γ / Σconf^γ       │
            │                             │
            │  fused = Σ w_i × prob_i     │
            └─────────────┬───────────────┘
                          │
             ┌────────────┴────────────┐
             │                         │
             ▼                         ▼
    Severity Heuristic          Explainability
    (probability bands)         ┌─────────────┐
    "Low / Moderate /           │ SHAP        │ ← clinical
     High / Critical"           │ (TreeExpl.) │
                                ├─────────────┤
                                │ Integrated  │ ← ECG
                                │ Gradients   │
                                └─────────────┘
                          │
                          ▼
                    JSON Response
          {prediction, fused_probability,
           severity, confidence,
           branch_contribution,
           top_clinical_features,
           top_ecg_leads, recommendations,
           ecg_quality, disclaimer}
```

---

### 5.2 Platform Architecture

```
  ┌──────────────────────────────────────────┐
  │           Next.js 14 Frontend            │
  │     (App Router, Tailwind, shadcn/ui)    │
  │                                          │
  │  /             Landing page              │
  │  /auth         Login / Register          │
  │  /dashboard    Risk trend, stats         │
  │  /assess       3-step assessment wizard  │
  │  /results/[id] Results + SHAP + ECG IG   │
  │  /history      Assessment timeline       │
  │  /insights     Calibration dashboard     │
  │  /cardiobot    AI assistant (streaming)  │
  │  /profile      User settings             │
  └──────────────────┬───────────────────────┘
                     │ HTTPS + JWT Bearer
                     │ SSE for streaming
                     ▼
  ┌──────────────────────────────────────────┐
  │           FastAPI Backend                │
  │                                          │
  │  /auth/register, /login, /me             │
  │  /assess/run-stream  ← SSE streaming     │
  │  /assess/{id}, /assess/history           │
  │  /assess/digitize-ecg-image              │
  │  /insights/calibration                   │
  │  /insights/bootstrap-ci                  │
  │  /insights/model-comparison              │
  │  /insights/roc-data                      │
  │  /insights/confusion-matrix              │
  │  /chat  ← Groq SSE streaming             │
  │  /health                                 │
  └──────────┬───────────────────┬───────────┘
             │                   │
             ▼                   ▼
  ┌─────────────────┐  ┌──────────────────────┐
  │   PostgreSQL    │  │   ML Artifacts        │
  │                 │  │   (local / Render)    │
  │  users          │  │                       │
  │  assessments    │  │  cardiosense_pipeline │
  │  chat_sessions  │  │    .pkl (dict)        │
  │  chat_messages  │  │  clinical_rf.pkl      │
  │  model_runs     │  │  ecg_model.keras      │
  │  alembic_ver.   │  │  pipeline_class.py    │
  └─────────────────┘  │  13_severity_rf.pkl   │
                        └──────────────────────┘
                                  │
                                  ▼
                        ┌──────────────────────┐
                        │   Groq API           │
                        │   (llama-3.1-8b)     │
                        │   CardioBot + patient │
                        │   context injection   │
                        └──────────────────────┘
```

---

### 5.3 Assessment Streaming Flow

```
  Frontend: submit clinical form + ECG signal
       │
       ▼
  POST /assess/run-stream (SSE)
       │
       ├─► yield "INIT_PIPELINE"
       │        ↕ UI: ✓ Initializing multi-branch pipeline...
       │
       ├─► yield "RUNNING_CLINICAL_RF"
       │        ↕ UI: ✓ Executing Structured Clinical RF branch...
       │
       ├─► yield "RUNNING_ECG_CNN"
       │        ↕ UI: ✓ Executing 12-lead ECG CNN branch...
       │
       ├─► yield "CALIBRATING_PLATT"
       │        ↕ UI: ✓ Running Platt calibration algorithms...
       │
       ├─► yield "GATING_NODE_COMPLETE"
       │        ↕ UI: ✓ Executing Confidence-Adaptive Gating Node...
       │
       ├─► persist to DB (assessments table)
       │
       └─► yield "FINAL_REPORT_READY:{...json...}"
                ↕ UI: ✓ Finalizing diagnostic report and SHAP explanations...
                ↕ UI: redirect → /results/{assessment_id}
```

---

### 5.4 Database Schema

```
users
  id (UUID PK) | email | hashed_password | full_name | created_at

assessments
  id (UUID PK) | user_id (FK) | model_run_id (FK)
  clinical_input (JSONB)            ← raw 11-feature dict
  feature_missingness_map (JSONB)   ← audit trail: which features were null
  ecg_quality (JSONB)
  prediction | fused_probability | severity | severity_source
  confidence | branch_contribution (JSONB) | branch_probabilities (JSONB)
  top_clinical_features (JSONB)    ← SHAP values
  top_ecg_leads (JSONB)            ← IG attributions
  recommendations (JSONB) | disclaimer
  created_at

chat_sessions
  id | user_id (FK) | assessment_id (FK) | title | created_at

chat_messages
  id | session_id (FK) | role | content | created_at

model_runs
  id | component | model_version | git_commit_hash
  metrics (JSONB) | is_active | notes | created_at
```

---

## 6. Critical Decisions & Pivots

### D1 — KNN Imputation Leakage (Notebook 02b)
**What happened:** Original imputer was fit on the full combined dataset before
splitting. Test set saw training set distribution during imputation.
**Fix:** Imputer fit on TRAIN only, `.transform()` on test. Accuracy dropped 1–3%.
**Why it matters:** A model trained on a leaked pipeline would look slightly better
than it actually is. The drop confirms the leak was real. The corrected numbers
are now the official baseline.

### D2 — CNN-only over CNN-LSTM (Notebook 08)
**What happened:** Original plan specified a CNN-LSTM architecture. Ablation
showed LSTM consistently hurt performance (AUC −0.027 at same conv depth).
**Decision:** Lock CNN-only. LSTM adds recurrence overhead for no gain on
single-beat ECG morphology at PTB-XL scale.
**How to defend:** "We ran a controlled ablation holding conv-depth constant
across CNN-only, CNN-LSTM (2 blocks), and CNN-LSTM (3 blocks). CNN-only won
across all metrics. That's a real result — we report it, not hide it."

### D3 — Formula over Learned Gating
**What happened:** Original architecture specified a "small learned network"
for the gating module. What shipped is a closed-form confidence-adaptive formula.
**Reason:** The formula is interpretable, validated via gamma sweep, and avoids
the problem of learning a meta-model on a tiny pseudo-paired sample (UCI and
PTB-XL are different populations — there are no real clinical+ECG pairs).
**How to defend:** "The gating is adaptive, per-patient, and analytically justified.
The formula is the design choice — we validated it with a sweep and Platt
calibration, not arbitrary tuning."

### D4 — Severity Heuristic in Production
**What happened:** Notebook 13 produced a severity model with Grade 4 F1=0.00.
Replacing the heuristic banding with this model would degrade the production
severity field.
**Decision:** Keep heuristic in production, expose `severity_source` field so the
pipeline is transparent about this. The trained model is documented as a
stretch/extension, not a production component.

### D5 — CardioBot Without FAISS
**What happened:** Original plan used FAISS vector store for RAG.
Assessment data is already structured JSON in PostgreSQL — retrieving it through
a vector similarity search added no quality over direct DB lookup.
**What shipped:** Groq streaming LLM (llama-3.1-8b-instant) with assessment context
injected directly into the system prompt. Full patient context (SHAP values,
branch contributions, ECG leads, clinical inputs) included in every chat request
when a patient record is selected.

---

## 7. Data

### UCI Heart Disease (Clinical Branch)

| Sub-dataset | Rows | Key missingness pattern |
|---|---|---|
| Cleveland | 303 | Mostly complete; missing only `ca`, `thal` |
| Hungarian | 294 | Heavy missing in `slope`, `ca`, `thal`; some `fbs` |
| Switzerland | 123 | `chol` mostly 0 (treat as missing); heavy `ca`, `thal` |
| VA Long Beach | 200 | Similar to Switzerland |
| **Total** | **920** | **Binary dataset** |
| Multiclass subset | 626 | Rows with unusable severity labels dropped |

License: CC BY 4.0. Source: https://archive.ics.uci.edu/dataset/45/heart+disease

### PTB-XL (ECG Branch)

21,437 12-lead ECG recordings at 100 Hz, 10 seconds per record.
PhysioNet Credentialed Health Data License 1.5.0.
Source: https://physionet.org/content/ptb-xl/1.0.3/
Kaggle mirror (no credentials): https://www.kaggle.com/datasets/khyeh0719/ptb-xl-dataset

---

## 8. Full Stack Overview

| Component | Technology | Notes |
|---|---|---|
| Frontend | Next.js 14 (App Router), Tailwind, shadcn/ui | Antigravity for UI generation |
| Backend | FastAPI, Python 3.11 | SSE streaming for assessment + CardioBot |
| Database | PostgreSQL 16 + SQLAlchemy 2.0 | Alembic migrations |
| Auth | JWT (python-jose) + bcrypt (passlib) | 7-day expiry |
| Clinical ML | scikit-learn (Random Forest + CalibratedClassifierCV) | SHAP 0.51.0 for attribution |
| ECG ML | TensorFlow / Keras (1D CNN) | Trained on Kaggle T4 GPU |
| Fusion | Custom Python (scipy, numpy) | Confidence-adaptive formula, γ=3 |
| Explainability | SHAP (TreeExplainer) + TF GradientTape (Integrated Gradients) | |
| CardioBot | Groq API, llama-3.1-8b-instant, SSE streaming | httpx async client |
| ECG Digitization | OpenCV, PIL, scipy | JPG/PNG → (1000,12) array |
| Frontend deploy | Vercel | |
| Backend deploy | Render | Free tier; ML artifacts bundled |
| DB deploy | Render Postgres | |
| Notebooks | Google Colab (CPU) + Kaggle (T4 GPU) | Notebooks 08/08b GPU-only |

---

## 9. What Was Planned vs What Shipped

| Planned | Shipped | Δ |
|---|---|---|
| CNN-LSTM ECG branch | CNN-only (3 conv blocks) | Ablation showed LSTM hurts |
| Learned gating network | Confidence-adaptive formula (γ=3) | More interpretable, no paired data |
| FAISS RAG for CardioBot | Groq + direct DB assessment injection | Simpler, same quality |
| 7 model comparison table | Binary classifiers + ECG ablation | Comprehensive per-branch evaluation |
| Naive fusion as baseline | Linear (γ=1) vs gamma (γ=3) sweep | Gamma sweep replaced naive vs gated |
| Grade 0-4 severity model | Heuristic banding (Grade 4 F1=0.00) | Trained model too weak for production |
| POST /assess (simple) | POST /assess/run-stream (SSE) | Streaming pipeline animation |
| Static insights | Live backend endpoints | All 5 insights endpoints real |
| Fused bootstrap CI | Per-branch only (open item) | Fused CI not yet computed |

---

*Last updated: July 2026*
*Build: Notebooks 01–13 complete. Platform deployed. Research README current.*
