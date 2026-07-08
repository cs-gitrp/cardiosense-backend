# CardioSense AI — Master Process Document

**Status: LOCKED. This is the final flagship project. Do not pivot further.**

This document is the single source of truth for this project. Use it with Claude or any other AI agent to maintain full context. Update it as decisions evolve, but the core architecture and scope below should not change.

---

## 1. Project Identity

**Name:** CardioSense AI
**Tagline:** A multimodal cardiac risk intelligence platform combining clinical data and ECG signals through a novel gated-fusion deep learning architecture.

**Type:** Full-stack, production-grade final year flagship project (BTech CSE — AI/ML), with a research paper as a secondary, non-blocking output.

**Primary goal:** A DocuMind-level (or better) deployed software platform, powered by genuine ML/DL technical depth — not a thin UI wrapper around a simple model, and not a paper-only research exercise with no usable product.

**Secondary goal:** If results are strong, package the ML core + experiments into an IEEE/Springer-style manuscript. Publication is NOT the design driver — system quality is. The paper, if pursued, is a byproduct of doing the ML work properly.

---

## 2. Layman Description

CardioSense looks at a patient's basic health numbers (age, blood pressure, cholesterol, ECG readings) and tells them — and a doctor — not just "you might have heart disease" but **how confident it is, which signals it trusted most for this specific patient, and how severe the risk is.** It reads a real ECG signal the way a cardiologist would, AND the clinical checkup numbers the way a GP would, then intelligently decides how much to trust each source for that particular patient — rather than treating all patients and all data sources the same way. Users get a personal dashboard, can track assessments over time, and can chat with an AI assistant (CardioBot) about their results and general heart health.

---

## 3. What Is Unique / Novel (the technical core)

Do not let this get diluted — this is the one architectural idea the whole project is built around.

### 3.1 Core novelty: Confidence-Gated Multimodal Fusion
- **Branch A (clinical/tabular):** feature-selected classical ML (RF/XGBoost) on UCI Heart Disease features.
- **Branch B (ECG signal):** CNN-LSTM hybrid on PTB-XL ECG signals.
- **Gated Fusion Module (the novel piece):** a small learned network that looks at how confident each branch is for THIS specific patient and weights the two predictions accordingly — not a fixed 50/50 average, which is what most "multimodal" student/research projects lazily do.

### 3.2 Secondary novelty: Missingness as a Feature, Not Just a Preprocessing Step
UCI Heart Disease has real missing values (`ca`, `thal` columns). Instead of silently imputing and moving on, missingness is treated as an explicit signal: missingness-indicator flags are fed into the model, and a dedicated experiment studies how prediction confidence and accuracy degrade as more clinical features go missing — and how the gating module compensates by leaning more on the ECG branch.

### 3.3 What is reused (not novel, by design)
- BERT/standard embeddings — N/A here (this is tabular+signal, not text)
- Dependency parsing — N/A
- Standard CNN/LSTM layers, standard RF/XGBoost — reused, established
- Standard feature selection algorithms (Chi-Square, LASSO, RF importance) — reused, but the *comparison* of all three against each other is part of the contribution

**One-sentence pitch for resume/interviews/abstract:**
"A confidence-gated dual-branch fusion model that combines feature-selected clinical data with ECG deep learning, learning per-patient trust weighting between modalities rather than fixed-weight fusion, with explicit missingness-aware handling of incomplete clinical records."

---

## 4. Datasets (Locked — exactly two, no more)

| Dataset | Purpose | Notes |
|---|---|---|
| **UCI Heart Disease** (Cleveland + combined 4-source version) | Clinical/tabular branch | ~920 rows combined. Real missingness in `ca`, `thal`. Supports both binary and original 0-4 multi-class severity grading — use multi-class, most projects lazily binarize. |
| **PTB-XL** (fallback: MIT-BIH) | ECG signal branch | 21,000+ labeled ECGs, multi-label diagnostic annotations, large enough to train a real CNN/LSTM. |

Do not add a third dataset "for completeness." Depth on two beats shallow coverage of three.

---

## 5. Model Comparison Plan (Existing vs. Proposed)

### 5.1 Models to implement and compare (7 total)
1. Logistic Regression (baseline)
2. Random Forest
3. SVM (RBF kernel)
4. XGBoost (strongest classical baseline expected)
5. CNN-LSTM on ECG only (no fusion) — DL-only baseline
6. Naive fusion (simple averaged tabular+ECG prediction, no learned gating) — "obvious" multimodal baseline
7. **Proposed: Dual-branch Gated Fusion model** (feature-selected RF/XGBoost + CNN-LSTM + learned gating)

### 5.2 Feature selection comparison (tabular branch)
Run and compare three methods, report selected features and downstream accuracy for each:
- Chi-Square
- LASSO (L1-regularized logistic regression)
- Random Forest Feature Importance

Train RF/XGBoost on each method's selected subset → compare accuracy/F1 → pick best-performing subset for the final pipeline. This comparison is itself a dashboard panel and a results-section table.

### 5.3 Class imbalance handling
- Tabular branch: **SMOTE-NC** (handles mixed categorical/continuous features) + class-weighted loss
- ECG branch: **Focal loss** for rarer arrhythmia classes in PTB-XL
- Multi-class severity grading (0-4) is more imbalanced than binary — use this as the primary imbalance story, not the binary version

### 5.4 Full metric set (report for ALL 7 models)
- Accuracy
- Precision
- Recall (Sensitivity) — flagged as most critical in medical context (false negatives = missed at-risk patients)
- F1-score
- Specificity
- ROC-AUC
- PR-AUC (more informative than ROC-AUC under imbalance)
- Confusion Matrix (full TP/TN/FP/FN, heatmap visualization per model)
- Cohen's Kappa / Quadratic Weighted Kappa (for multi-class severity grading)
- **5-fold or 10-fold cross-validation** for every model — report mean ± std, not single train/test split numbers

### 5.5 Ablation studies (mandatory)
- Gating ON vs OFF (fixed-weight fusion)
- Tabular-only vs ECG-only vs full dual-branch
- With vs without missingness-indicator flags
- Multi-class severity vs binarized version

### 5.6 Presentation structure (powers both the in-app dashboard AND paper results section)
1. Main comparison table — 7 models × all metrics, proposed model row highlighted, best score per column bolded
2. Confusion matrix grid — one heatmap per model
3. Overlaid ROC curves — all models, one plot
4. Feature selection comparison table
5. Ablation table
6. Bar chart — accuracy/F1/recall across all 7 models

---

## 6. System Scope Decision

**Build a full platform. Do not compromise on this.** Auth, dashboard, history, chatbot, and an in-app evaluation/insights view are all in scope and required — this is what makes it "DocuMind-level or better," not just a notebook with a model.

**What is explicitly in scope:**
- Full auth (signup/login)
- Personal dashboard with assessment history and risk trend over time
- New assessment flow (clinical form + ECG upload)
- Results page with explainability (branch attribution)
- Chatbot (RAG-based, grounded in user's own data + cardiology knowledge base)
- Model Insights / Evaluation dashboard showing the full Section 5 comparison live in-product
- Settings/profile

**What is explicitly out of scope (no future-scope creep):**
- Multi-disease support (no diabetes/cancer modules — single disease, full depth, as already decided)
- Multilingual support
- Real-time streaming data ingestion
- Third dataset
- Mobile app (web only)

---

## 7. PRD (Product Requirements)

**Problem statement:** Existing cardiac risk prediction systems rely on either clinical data alone or ECG signals alone, and when they combine both, they use naive fixed-weight fusion — failing to account for per-patient data quality/completeness differences that should change how much each modality is trusted.

**Target users:** Demonstration/portfolio context — framed as a tool for clinicians/patients to get a risk assessment with transparency into *why* the model made its decision and which data source it trusted more.

**Core features (full list, build all of these — no shallow "decide later"):**
1. User authentication (signup/login, JWT-based)
2. New Assessment: form for clinical features (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal) + ECG file upload or sample selection
3. Results page: risk score, severity grade (0-4), branch-attribution explanation, downloadable report
4. Assessment History: list of past assessments per user, searchable/filterable, risk trend over time
5. CardioBot chatbot: RAG-grounded, answers questions about the user's own result and general cardiology knowledge; explicitly informational only, with a visible medical disclaimer (not a diagnostic tool)
6. Model Insights / Evaluation Dashboard: full Section 5 comparison tables and charts, live in the product
7. Settings/profile management

---

## 8. TRD (Technical Requirements)

**ML Stack:**
- Tabular models: scikit-learn (Logistic Regression, RF, SVM), XGBoost
- Deep learning: PyTorch (CNN-LSTM for ECG)
- Feature selection: scikit-learn (Chi-Square, LASSO via `Lasso`/`LogisticRegression(penalty='l1')`, RF importance)
- Imbalance handling: `imbalanced-learn` (SMOTE-NC), custom focal loss implementation in PyTorch
- ECG signal processing: `wfdb` (for PTB-XL), `scipy`/`neurokit2` for signal preprocessing/denoising

**Backend:** FastAPI + PostgreSQL + SQLAlchemy (same proven stack as DocuMind)
**Frontend:** Next.js + Tailwind + shadcn/ui (same proven stack as DocuMind)
**Chatbot/RAG:** Same architecture pattern as DocuMind — sentence-transformers embeddings + FAISS vector store + Groq API (llama-3.1-8b-instant) or equivalent for generation
**Model serving:** Model checkpoints loaded directly into the FastAPI process at startup — no separate HF Endpoint needed, model is lightweight enough
**Auth:** JWT-based, bcrypt password hashing

**Database tables:** `users`, `assessments` (clinical inputs + ECG reference + results + branch confidences), `chat_sessions`, `chat_messages`, `model_runs` (stores metrics for the Insights dashboard)

---

## 9. System Architecture

### 9.1 ML Core Architecture

```
Clinical Features (tabular)          ECG Signal (raw waveform)
        │                                     │
        ▼                                     ▼
Missingness-aware                      Preprocessing
preprocessing                          (denoise, segment,
(imputation + missingness               normalize)
 indicator flags)                            │
        │                                    ▼
        ▼                            CNN-LSTM feature
Feature Selection                     extractor
(Chi-Sq / LASSO / RF                        │
 importance comparison)                     ▼
        │                            ECG risk score +
        ▼                            confidence estimate
Classical ML                                │
(RF / XGBoost,                              │
 SMOTE-NC + class-weighted)                 │
        │                                   │
        ▼                                   │
Tabular risk score +                        │
confidence estimate                         │
        │                                   │
        └───────────────┬───────────────────┘
                         ▼
              Gated Fusion Module
        (learns per-patient trust weight
         between tabular vs ECG branch)
                         │
                         ▼
              Final Risk Prediction
              + Severity Grade (0-4)
              + Branch Attribution Explanation
```

### 9.2 Full System Architecture (Platform Level)

```
                    ┌─────────────────────────┐
                    │   Next.js Frontend       │
                    │  (Tailwind + shadcn/ui)  │
                    └───────────┬──────────────┘
                                │ REST API (JWT auth)
                    ┌───────────▼──────────────┐
                    │     FastAPI Backend       │
                    ├───────────────────────────┤
                    │ Auth Service               │
                    │ Assessment Service ───────┼──► ML Core (Section 9.1)
                    │ Chatbot Service ──────────┼──► FAISS Vector Store
                    │                            │    + Groq API
                    │ Insights Service           │
                    └───────────┬───────────────┘
                                │
                    ┌───────────▼──────────────┐
                    │   PostgreSQL Database     │
                    │ users, assessments,       │
                    │ chat_sessions/messages,    │
                    │ model_runs                 │
                    └────────────────────────────┘
```

### 9.3 API Endpoints

- `POST /auth/signup`, `POST /auth/login`
- `POST /assess` — clinical features + ECG → dual-branch model → risk score, severity, branch-confidence breakdown
- `GET /assess/{id}` — retrieve a past assessment
- `GET /history` — user's assessment history
- `POST /chatbot` — RAG-grounded chat (assessment context + cardiology knowledge base)
- `GET /insights/model-comparison` — Section 5 full comparison data
- `GET /insights/feature-selection` — Chi-Sq/LASSO/RF importance comparison data
- `GET /insights/ablation` — gating on/off, branch-only, missingness-flag ablation data

---

## 10. Frontend Pages / UI

1. **Landing / Login / Signup**
2. **Dashboard** — past assessments, risk trend chart, quick stats
3. **New Assessment** — clinical feature form + ECG upload/sample selection
4. **Results Page** — risk score, severity grade, branch-attribution explanation, downloadable report
5. **Assessment History** — searchable/filterable list, trend view
6. **CardioBot (Chatbot)** — conversational interface, grounded in user's assessment + cardiology knowledge base, visible disclaimer banner
7. **Model Insights / Evaluation Dashboard** — full model comparison table, confusion matrix grid, ROC curves, feature selection comparison, ablation table, bar charts
8. **Settings** — profile, data management

---

## 11. Prerequisites Before Writing Code

- [ ] Read 2-3 reference papers on multimodal fusion for clinical+signal data (to properly cite "naive fusion" as the weak baseline you're beating)
- [ ] Read 1-2 reference papers on PTB-XL CNN/LSTM classification approaches (architecture reference for ECG branch)
- [ ] Get comfortable with `wfdb` library for reading PTB-XL ECG records
- [ ] Get comfortable with `imbalanced-learn`'s SMOTE-NC for mixed-type tabular data
- [ ] Set up GPU environment (Colab Pro or Kaggle GPU — T4/P100 sufficient for CNN-LSTM on ECG; tabular models run fine on CPU)
- [ ] Review your own DocuMind RAG pipeline code — the chatbot here reuses that pattern almost directly

---

## 12. Free AI Tools / Software — Exact Purpose

| Tool | Purpose |
|---|---|
| **Claude (this chat)** | Architecture decisions, debugging, code review, paper/report writing assistance |
| **GitHub Copilot (free student pack)** | Daily coding driver — boilerplate, FastAPI routes, React components |
| **v0.dev** | UI component generation for frontend pages |
| **Google Colab (free/Pro tier)** | GPU training for CNN-LSTM ECG model and any DL experiments |
| **Kaggle Notebooks** | Alternative free GPU environment, also where PTB-XL/UCI datasets are easily accessible |
| **Hugging Face (free tier)** | Hosting sentence-transformers embedding model for chatbot RAG (if not run locally) |
| **Groq API (free tier)** | LLM inference for CardioBot chatbot generation (same as DocuMind setup) |
| **scikit-learn, XGBoost, imbalanced-learn** | Classical ML models, feature selection, SMOTE-NC — all free, open-source |
| **PyTorch** | CNN-LSTM ECG model, gated fusion module implementation |
| **wfdb (PhysioNet toolkit)** | Reading/parsing PTB-XL ECG signal data |
| **neurokit2 / scipy** | ECG signal preprocessing, denoising |
| **FAISS** | Vector store for chatbot RAG (same as DocuMind) |
| **Vercel (free tier)** | Frontend deployment |
| **Render (free tier)** | Backend + PostgreSQL deployment |
| **draw.io / Mermaid (free)** | Architecture diagrams for README and paper figures |
| **Overleaf (free tier)** | LaTeX paper writing if pursuing publication |
| **Google Scholar / Connected Papers (free)** | Literature review, finding related work for paper positioning |

---

## 13. Exact Roadmap (Week-by-Week)

| Phase | Weeks | Deliverable |
|---|---|---|
| 1. Setup + literature skim + data acquisition | 1 | Repos created, both datasets downloaded, 2-3 reference papers read |
| 2. Data preprocessing (both branches) | 1 | Cleaned tabular data with missingness flags, preprocessed/segmented ECG signals, train/test splits |
| 3. Feature selection comparison + classical ML baselines | 1 | Chi-Sq/LASSO/RF importance comparison done, models 1-4 (LogReg, RF, SVM, XGBoost) trained and evaluated |
| 4. ECG branch (CNN-LSTM) | 1.5 | Model 5 (ECG-only) trained, evaluated |
| 5. Naive fusion baseline + Gated Fusion model (the core novel work) | 2 | Models 6 and 7 built, trained, evaluated |
| 6. Full evaluation suite (Section 5.4, 5.5) | 1 | All metrics, confusion matrices, ROC curves, ablations complete for all 7 models |
| 7. Backend (auth, assessment service, insights service) | 1.5 | Working API, all endpoints functional |
| 8. Chatbot (RAG) | 0.5 | CardioBot functional, reusing DocuMind RAG pattern |
| 9. Frontend (all 8 pages) | 2 | Fully functional UI, polished to DocuMind standard or better |
| 10. Deployment + documentation | 1 | Deployed app, README, EVAL_REPORT, screenshots, resume bullets |
| 11. (Optional, parallel from week 5 onward) Paper writing | ongoing | Draft manuscript, submission-ready if pursuing publication |

**Total: ~12-13 weeks for full build.** Paper writing runs in parallel, not sequentially after.

---

## 14. Step-by-Step Starting Guidance (Exact First Steps)

1. **Today:** Create two repos — `cardiosense-ai` (frontend) and `cardiosense-backend` (backend), mirroring DocuMind's folder structure.
2. **Today/tomorrow:** Download UCI Heart Disease (combined 4-source CSV) and set up PTB-XL access (via `wfdb` + PhysioNet, or Kaggle's PTB-XL mirror).
3. **This week:** Open the UCI data, inspect missingness in `ca`/`thal` manually — decide exact imputation strategy before writing pipeline code. Open a handful of PTB-XL ECG records with `wfdb`, visualize a few waveforms, understand the signal format before building preprocessing code.
4. **This week:** Read 1 ECG-CNN reference paper and 1 multimodal clinical+signal fusion reference paper — these anchor your "naive fusion" baseline definition and your gated fusion's positioning.
5. **End of week 1:** Have both datasets cleaned, split, and ready; have a literature note (2-3 papers: method, dataset, key result) to reference later for the paper and for justifying your baseline choices.

---

## 15. Working Agreement / Notes for Future Sessions

- This document is the locked source of truth. Do not propose scope pivots — bring questions about *implementation* of what's already decided here.
- If stuck on any phase, reference the relevant section number above when asking Claude or another agent for help, to preserve context without re-explaining the whole project.
- Update Section 13 (roadmap) checkboxes/status as phases complete, so future sessions know exactly where the project stands.
