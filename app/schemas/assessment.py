from pydantic import BaseModel, Field
from typing import Optional
import uuid


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class ClinicalFeatures(BaseModel):
    """
    The 11 consensus-selected features from 04_selected_features.csv.
    Field order matches manifest.json clinical_features list exactly.
    Missing values should be submitted as None — the inference pipeline
    handles them via the same imputation path used in training
    (fit on train-only as per Notebook 02b fix).

    Encoding reference (for frontend form):
      cp:           chest pain type  (0=typical angina, 1=atypical, 2=non-anginal, 3=asymptomatic)
      exang:        exercise-induced angina (0=No, 1=Yes)
      ca:           number of major vessels coloured by fluoroscopy (0-3); None if unknown
      sex:          0=Female, 1=Male
      chol_missing: 1 if cholesterol was 0/missing in source record, else 0
      slope_missing:1 if slope was missing in source record, else 0
      thal:         thalassemia (1=normal, 2=fixed defect, 3=reversable defect); None if unknown
      thalach:      maximum heart rate achieved
      age:          age in years
      oldpeak:      ST depression induced by exercise relative to rest
      fbs:          fasting blood sugar > 120 mg/dl (0=No, 1=Yes)
    """
    cp: float
    exang: float
    ca: Optional[float] = None
    sex: float
    chol_missing: float
    slope_missing: float
    thal: Optional[float] = None
    thalach: float
    age: float
    oldpeak: float
    fbs: float


class AssessRequest(BaseModel):
    clinical: ClinicalFeatures
    # ECG is sent as a flat list (1000 samples × 12 leads = 12000 floats),
    # reshaped to (1000, 12) numpy array in the service layer before
    # calling pipeline.run(). Optional: assessment can run clinical-only.
    ecg_signal: Optional[list[float]] = Field(
        default=None,
        description="Flat array of 12000 floats, pre-processed (1000×12 leads). "
                    "If omitted, fusion falls back to clinical-only mode."
    )


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------

class BranchContribution(BaseModel):
    clinical_pct: float
    ecg_pct: float


class BranchProbabilities(BaseModel):
    clinical: float
    ecg: Optional[float] = None


class AssessResponse(BaseModel):
    assessment_id: str
    prediction: str                         # "Disease" | "No Disease"
    fused_probability: float
    severity: Optional[str] = None          # "Low" | "Moderate" | "High" | "Critical"
    severity_source: Optional[str] = None   # "heuristic_probability_band" |
                                             # "trained_multiclass_rf"
    confidence: float
    branch_contribution: Optional[BranchContribution] = None
    branch_probabilities: Optional[BranchProbabilities] = None
    top_clinical_features: Optional[list] = None
    top_ecg_leads: Optional[list] = None
    ecg_quality: Optional[dict] = None
    recommendations: Optional[list] = None
    feature_missingness_map: Optional[dict] = None   # Phase 7 audit trail
    disclaimer: str
    clinical: Optional[ClinicalFeatures] = None       # original patient input features
    created_at: Optional[str] = None                  # ISO timestamp of assessment
    doctor_summary: str | None = None
    patient_summary: str | None = None
    lead_annotation: str | None = None
    display_waveforms: dict[str, str] | None = None

class AssessmentHistoryItem(BaseModel):
    assessment_id: str
    prediction: str
    fused_probability: float
    severity: Optional[str]
    created_at: str

    class Config:
        from_attributes = True
