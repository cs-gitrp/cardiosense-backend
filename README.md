<div align="center">

# CardioSense AI — Backend

**Production FastAPI backend and research ML pipeline for multimodal cardiac risk assessment**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-cardiosense.vercel.app-0ea5e9?style=flat-square&logo=vercel)](https://cardiosense-gamma.vercel.app/)
[![API Docs](https://img.shields.io/badge/API%20Docs-Swagger%20UI-009688?style=flat-square&logo=fastapi)](https://cardiosense-backend-7j16.onrender.com/docs)
[![Frontend](https://img.shields.io/badge/Frontend%20Repo-cardiosense--frontend-6366f1?style=flat-square&logo=github)](https://github.com/cs-gitrp/cardiosense-frontend)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)

</div>

---

## What this system does

CardioSense AI is a **clinical decision support research prototype** that fuses two independent diagnostic branches — a 1D-CNN trained on 12-lead ECG signals (PTB-XL, 21,837 records) and a Random Forest trained on structured clinical observations (UCI Heart Disease) — into a single calibrated risk prediction using confidence-adaptive weighting.

Three design decisions distinguish it from a standard ML pipeline:

**1. Confidence-adaptive fusion** — The fusion layer does not use fixed weights. It proportionally trusts whichever branch is more confident for a given patient, implemented with gamma=3 Platt-calibrated sharpening selected via sensitivity sweep. When ECG quality is high, the CNN dominates. When clinical data is more complete, the RF dominates.

**2. Calibrated probabilities** — Most models output a score. Calibrated probabilities mean something: when this system outputs 70% risk, approximately 70 in 100 similar patients have the condition. ECE (Expected Calibration Error) of **0.034** on the ECG branch — verified via 1,000-sample bootstrap.

**3. Full explainability** — SHAP attributions on the RF branch identify which clinical features drove the prediction. Integrated Gradients on the CNN highlight which leads and waveform regions contributed most. Both are surfaced in the API response and rendered in the frontend.

---

## Model Performance

Performance metrics are reported on independently held-out test sets.
Bootstrap confidence intervals were computed using **1000 bootstrap resamples**.

| Branch | AUC | 95% CI | Brier Score | ECE |
|--------|------:|:---------------:|------:|------:|
| ECG (CNN) | **0.9424** | 0.9347 – 0.9504 | 0.0943 | 0.034 |
| Clinical (RF) | **0.9266** | 0.8887 – 0.9582 | 0.1130 | *See calibration analysis* |

> **Fusion module:** CardioSense uses a **confidence-adaptive decision-level fusion** mechanism built on calibrated branch probabilities. Because the clinical (UCI Heart Disease) and ECG (PTB-XL) datasets originate from different patient cohorts and are **not patient-paired**, the fusion component is validated as an **architectural decision module** rather than through a combined held-out accuracy metric. Therefore, **no aggregate Fusion AUC is reported**.

Clinical branch recall: **90.2%** — intentionally optimized to minimize false negatives, which are more critical in cardiac risk screening than false positives.

**Key design decisions contributing to these results:**

- Patient-grouped **GroupShuffleSplit** on `patient_id` eliminated patient leakage in PTB-XL by ensuring that no patient's ECGs appeared across multiple splits.
- Confidence calibration using **Platt Scaling** before fusion improved probability reliability, reducing ECG Expected Calibration Error (ECE) to **0.034**.
- Clinical missing values were modeled as explicit information through engineered missingness indicators instead of being discarded or blindly imputed.
- Three ECG architectures were evaluated during ablation. A **3-block CNN** consistently outperformed both CNN-LSTM variants and was selected as the final ECG architecture.
- Confidence-adaptive fusion dynamically adjusts branch influence according to calibrated prediction confidence, replacing conventional fixed-weight multimodal fusion.
- Model explanations combine **SHAP** (clinical branch) and **Integrated Gradients** (ECG branch) to provide modality-specific interpretability.
- Performance uncertainty is quantified using **1000-bootstrap confidence intervals** for robust evaluation rather than relying solely on point estimates.

---

## System architecture

```mermaid
flowchart TD
    subgraph CLIENT["Client Layer"]
        FE["Next.js 14 Frontend\ncardiosense.vercel.app"]
    end

    subgraph API["FastAPI Application  ·  Granian ASGI  ·  Render"]
        AUTH["POST /auth/register\nPOST /auth/login\nGET  /auth/me\n─────────────────\nJWT (bcrypt + HS256)"]
        
        STREAM["POST /assess/run-stream\n─────────────────\nServer-Sent Events\n6 stage tokens"]
        
        HISTORY["GET  /assess/history\nGET  /assess/{id}\nDELETE /assess/{id}"]
        
        ECG_UPLOAD["POST /assess/digitize-ecg-image\n─────────────────\nECG image → waveform\nvector digitization service"]
        
        INSIGHTS["GET /insights/calibration\nGET /insights/bootstrap-ci\nGET /insights/model-comparison\nGET /insights/roc-data\nGET /insights/confusion-matrix"]
        
        CHAT["POST /chat\n─────────────────\nSSE streaming\nGroq context injection"]
    end

    subgraph ML["ML Inference Layer  ·  ml_artifacts/"]
        direction TB

        subgraph BRANCH_A["Clinical Branch"]
            RF["Random Forest\nsklearn · UCI Heart Disease\n303 records · 11 features"]
            SHAP["SHAP Attributions\nfeature importance\nper prediction"]
            PLATT_C["Platt Calibration\ngamma = 3"]
            RF --> PLATT_C
            RF --> SHAP
        end

        subgraph BRANCH_B["ECG Branch"]
            CNN["1D-CNN\nTensorFlow / Keras\nPTB-XL · 21,837 records\n1000 samples × 12 leads"]
            IG["Integrated Gradients\nlead + waveform\nattribution"]
            PLATT_E["Platt Calibration\ngamma = 3"]
            CNN --> PLATT_E
            CNN --> IG
        end

        FUSION["Confidence-Adaptive Fusion\n─────────────────────────\nw_ecg  = softmax(conf_ecg  ^ γ)\nw_clin = softmax(conf_clin ^ γ)\np_fused = w_ecg·p_ecg + w_clin·p_clin"]

        PLATT_C --> FUSION
        PLATT_E --> FUSION

        RISK["Risk Grade\nLow · Moderate · High · Critical\n─────────────────\nthreshold-gated from p_fused"]

        FUSION --> RISK
    end

    subgraph BOT["CardioBot  ·  Groq API"]
        GROQ["llama-3.1-8b-instant\n─────────────────\nFull assessment history\ninjected as context\n(DB lookup, not RAG)"]
    end

    subgraph DB["Supabase PostgreSQL  ·  ap-south-1"]
        USERS["users"]
        ASSESS["assessments\n+ shap_values JSON\n+ ig_values JSON"]
        CHAT_HIST["chat_messages"]
        MODEL_META["model_runs"]
    end

    FE -->|"HTTPS + Bearer JWT"| AUTH
    FE -->|"multipart/form-data"| ECG_UPLOAD
    FE -->|"SSE  text/event-stream"| STREAM
    FE -->|"REST"| HISTORY
    FE -->|"REST"| INSIGHTS
    FE -->|"SSE  text/event-stream"| CHAT

    STREAM --> ML
    HISTORY --> DB
    INSIGHTS --> MODEL_META

    RISK --> DB
    SHAP --> DB
    IG --> DB

    CHAT --> GROQ
    GROQ -->|"DB lookup\nassessment context"| DB

    AUTH --> USERS
```

---

## Streaming inference — how `POST /assess/run-stream` works

This endpoint is a **Server-Sent Events stream**, not a standard JSON response. It drives the animated 6-stage pipeline in the frontend.

```
Stage 1 → ECG_PROCESSING        CNN forward pass on 1000×12 signal
Stage 2 → CLINICAL_ANALYSIS     RF inference on 11 clinical features
Stage 3 → FEATURE_EXTRACTION    SHAP + IG attribution computation
Stage 4 → RISK_FUSION           Confidence-adaptive weighting applied
Stage 5 → CALIBRATION           Platt scaling (gamma=3) on fused probability
Stage 6 → ASSESSMENT_COMPLETE   Full JSON payload flushed — risk grade, probs, attributions
```

Each stage token is yielded as `data: {"stage": "ECG_PROCESSING", "status": "running"}` over the SSE connection. The frontend advances its pipeline animation on each token. The final `ASSESSMENT_COMPLETE` token carries the complete inference result, which is then persisted to Supabase and rendered in the UI.

**Why SSE and not WebSocket?** Inference is unidirectional server→client. SSE is simpler to proxy on Render and does not require a persistent bidirectional connection.

---

## CardioBot — context injection architecture

CardioBot is **not a generic RAG chatbot**. It does not embed queries against a vector index.

On every request to `POST /chat`, the backend:
1. Authenticates the JWT and resolves the user
2. Fetches the user's last N assessments from Supabase (risk grades, probabilities, SHAP attributions, timestamps)
3. Serializes the full assessment context as a structured system prompt
4. Sends to Groq (`llama-3.1-8b-instant`) with the user's message
5. Streams the response back via SSE

This means every CardioBot response is grounded in the actual patient's assessment history — not generic cardiac knowledge. The model has no access to other users' data.

---

## Repository structure

```
cardiosense-backend/
│
├── app/                          Production FastAPI application
│   ├── api/                      Route handlers (auth, assessment, insights, chat, ecg)
│   ├── core/                     Config, security (JWT, bcrypt)
│   ├── db/                       SQLAlchemy session, Alembic migrations
│   ├── models/                   ORM models (users, assessments, chat, model_runs)
│   ├── schemas/                  Pydantic v2 request/response schemas
│   ├── services/                 Business logic (assessment, model loader, ECG digitizer)
│   └── main.py                   App entrypoint, lifespan, router registration
│
├── ml_artifacts/                 Frozen production inference artifacts (binary)
│   ├── cardiosense_pipeline/     Clinical RF + ECG CNN + fusion pipeline
│   └── severity/                 Notebook 13 multiclass RF (secondary, not prod)
│
├── research/                     Complete ML research pipeline
│   ├── data/                     Raw datasets + download instructions
│   ├── notebooks/                Notebooks 01–13 in execution order
│   ├── results/                  Metrics, bootstrap CI plots, ROC curves
│   ├── process.md                Full research decisions and rationale log
│   └── research_README.md        Research pipeline documentation
│
├── sample_data/                  De-identified demo records for local testing
├── diagnose_pipeline.py          Artifact integrity check + inference smoke test
├── alembic.ini
├── requirements.txt
├── runtime.txt
└── .env.example
```

---

## ML Research Pipeline (14 Development Notebooks)

The `research/notebooks/` directory contains the complete reproducible ML development pipeline behind CardioSense AI. Each notebook represents a major milestone in the research workflow and should be executed sequentially.

| Notebook | Purpose | Key Output |
|---|---|---|
| `01_data_exploration.ipynb` | Exploratory analysis of the UCI Heart Disease dataset, feature distributions, missing values and class balance | Dataset statistics & EDA visualizations |
| `02_proper_preprocessing.ipynb` | Clinical preprocessing, missing-value handling, feature engineering and leakage-safe dataset preparation | Cleaned training dataset |
| `03_baseline_models_v1.ipynb` | Baseline machine learning experiments (Logistic Regression, SVM, Decision Tree, Random Forest, XGBoost) | Baseline performance comparison |
| `04_feature_selection.ipynb` | Feature selection using Chi-Square, LASSO and Random Forest importance | Ranked feature importance |
| `05_feature_selection_validation.ipynb` | Validation of reduced-feature models against full-feature models | Final clinical feature subset |
| `06_ptbxl_metadata_preparation.ipynb` | PTB-XL metadata parsing, diagnostic superclass mapping and dataset preparation | Processed PTB-XL metadata |
| `07_ecg_signal_processing.ipynb` | ECG waveform loading, normalization, patient-wise splitting and preprocessing | Model-ready ECG tensors |
| `08_cnn_lstm_ecg_classifier.ipynb` | CNN and CNN-LSTM architecture experiments for ECG classification | Architecture comparison results |
| `08b_final_ecg_training.ipynb` | Final production 3-block CNN training, evaluation and model export | Production ECG model |
| `09_confidence-adaptive-fusion.ipynb` | Platt calibration, confidence-adaptive fusion and probability calibration | Fusion inference pipeline |
| `10_explainable-ai.ipynb` | SHAP explanations for the clinical branch and Integrated Gradients for ECG interpretation | Explainability artifacts |
| `11_comprehensive-evaluation.ipynb` | ROC analysis, confusion matrix, calibration curves and bootstrap confidence intervals | Final evaluation metrics |
| `12_cardiosense-inference-pipeline.ipynb` | End-to-end production inference pipeline and artifact packaging | Production inference pipeline |
| `13_multiclass-severity-model.ipynb` | Experimental multiclass severity prediction (Low–Critical); secondary research module not used in production inference | Severity prediction model |

**Key research decisions documented throughout the notebook pipeline:**

- `02` — Clinical preprocessing removes data leakage, engineers missing-value indicators and prepares production-ready feature pipelines.
- `05` — Feature-selection validation confirms that the reduced clinical feature set preserves predictive performance while simplifying inference.
- `07` — Patient-wise splitting (`GroupShuffleSplit`) prevents ECG data leakage across train, validation and test sets.
- `08 / 08b` — Multiple ECG architectures were evaluated. The final deployed model is the optimized 3-block CNN selected after CNN-LSTM ablation experiments.
- `09` — Confidence-adaptive multimodal fusion combines calibrated branch probabilities using Platt scaling with an experimentally selected gamma value.
- `10` — Explainability integrates SHAP for structured clinical features and Integrated Gradients for ECG waveform attribution.
- `11` — Comprehensive evaluation reports ROC curves, calibration diagnostics, confusion matrices and bootstrap confidence intervals.
- `12` — The complete production inference contract freezes preprocessing, model loading, fusion logic and output schema used by the deployed API.
- `13` — Multiclass severity prediction is retained as an experimental extension for future work and is intentionally excluded from the production inference pipeline.

---

## API reference

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| POST | `/auth/register` | — | Register user, returns JWT |
| POST | `/auth/login` | — | Login, returns JWT |
| GET | `/auth/me` | Bearer | Current user profile |
| POST | `/assess/run-stream` | Bearer | **Streaming SSE inference** — 6 stage tokens + final payload |
| GET | `/assess/history` | Bearer | Paginated assessment history |
| GET | `/assess/{id}` | Bearer | Single assessment with SHAP + IG detail |
| DELETE | `/assess/{id}` | Bearer | Delete assessment |
| POST | `/assess/digitize-ecg-image` | Bearer | ECG image → waveform digitization |
| GET | `/insights/calibration` | Bearer | Calibration curve data points |
| GET | `/insights/bootstrap-ci` | Bearer | Bootstrap CI (n=1000) per branch |
| GET | `/insights/model-comparison` | Bearer | Ablation table: RF / CNN / Fused |
| GET | `/insights/roc-data` | Bearer | ROC curve points, all three branches |
| GET | `/insights/confusion-matrix` | Bearer | Confusion matrix at decision threshold |
| POST | `/chat` | Bearer | CardioBot — SSE streaming, Groq context injection |
| GET | `/health` | — | Health check |

Full interactive docs: `https://cardiosense-backend-7j16.onrender.com/docs`

---

## Local development

**Prerequisites:** Python 3.11, PostgreSQL 16

```bash
git clone https://github.com/cs-gitrp/cardiosense-backend
cd cardiosense-backend

python -m venv env
source env/bin/activate          # Linux/Mac
# env\Scripts\activate           # Windows

pip install -r requirements.txt
```

Copy and fill `.env.example`:

```env
DATABASE_URL=postgresql://cardiosense:password@localhost:5432/cardiosense
JWT_SECRET_KEY=<python -c "import secrets; print(secrets.token_hex(32))">
MODEL_DIR=./ml_artifacts
GROQ_API_KEY=<free key at console.groq.com>
```

Run migrations:

```bash
alembic upgrade head
```

**ML artifacts** — binary files are not in the repository. Either:
- Reproduce from Notebook 12 (`research/notebooks/12_cardiosense-inference-pipeline.ipynb`)
- Contact for access to the serialized artifacts

Start the server:

```bash
uvicorn app.main:app --reload
# API:     http://localhost:8000
# Swagger: http://localhost:8000/docs
```

---

## Deployment

| Layer | Platform | Notes |
|---|---|---|
| Backend | Render (free tier) | Granian ASGI server; cold start ~30–60s after inactivity |
| Database | Supabase PostgreSQL | ap-south-1 region; Alembic migrations applied on deploy |
| Frontend | Vercel | Separate repo — [cardiosense-frontend](https://github.com/cs-gitrp/cardiosense-frontend) |

Environment variables set in Render dashboard. `MODEL_DIR` points to the serialized artifacts loaded at startup via `model_loader.py` lifespan handler.

---

## Related repositories

| Repository | Description |
|---|---|
| [cardiosense-frontend](https://github.com/cs-gitrp/cardiosense-frontend) | Next.js 14 · TypeScript · Tailwind · shadcn/ui · Recharts |

---

## Author

**Chandan Singh** · B.Tech CSE (AI/ML)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-chandan--singh-0077b5?style=flat-square&logo=linkedin)](https://linkedin.com/in/chandan-singh-a23563304)
[![GitHub](https://img.shields.io/badge/GitHub-cs--gitrp-181717?style=flat-square&logo=github)](https://github.com/cs-gitrp)
[![Credly](https://img.shields.io/badge/Credly-Certifications-ff6a00?style=flat-square&logo=credly)](https://credly.com/users/chandan-singh.f55fc216)

---

## Disclaimer

CardioSense AI is a research prototype. It is not a medical device and has not been clinically validated. It must not be used for diagnosis or treatment decisions. All outputs require review by a licensed healthcare professional.
