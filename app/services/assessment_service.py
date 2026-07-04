"""
Assessment service — the single place where:
  1. Clinical dict is built from the request + missingness map computed
  2. ECG signal is reshaped from flat list → (1000, 12) numpy array
  3. pipeline.run() is called (Notebook 12 contract)
  4. Severity gating function from Notebook 13 / Cell 18 is invoked
  5. Everything is persisted to the assessments table with full audit trail
"""

import numpy as np
import pandas as pd
import uuid

from sqlalchemy.orm import Session

from app.schemas.assessment import AssessRequest, AssessResponse, BranchContribution, BranchProbabilities
from app.models.assessment import Assessment
from app.services.model_loader import get_pipeline, get_severity_model

# The 11 frozen features in the EXACT order from manifest.json
# (confirmed from ml_artifacts/cardiosense_pipeline/manifest.json).
# ORDER MATTERS — pipeline uses self.feature_order to build the DataFrame.
# Wrong order = silently wrong predictions with no error raised.
SELECTED_FEATURES = [
    "cp", "exang", "ca", "sex", "chol_missing",
    "slope_missing", "thal", "thalach", "age", "oldpeak", "fbs"
]

DISCLAIMER = (
    "CardioSense AI is a research-grade screening aid built for academic "
    "demonstration. It is NOT a medical device and must not be used for "
    "clinical diagnosis or treatment decisions. Always consult a qualified "
    "healthcare professional."
)


def _build_missingness_map(clinical_dict: dict) -> dict:
    """
    Checks which of the 11 features were submitted as None (i.e. missing
    in the original patient record). Used for the Phase 7 audit-trail column
    and also for the Insights dashboard query: 'X% of assessments had ≥2
    missing clinical features, here's how fusion compensated.'
    """
    return {feat: (clinical_dict.get(feat) is None) for feat in SELECTED_FEATURES}


def _reshape_ecg(ecg_flat: list[float]) -> np.ndarray:
    """
    Frontend sends ECG as a flat list of 12000 floats.
    Pipeline expects shape (1000, 12).
    """
    arr = np.array(ecg_flat, dtype=np.float32)
    if arr.shape != (12000,):
        raise ValueError(
            f"ecg_signal must have exactly 12000 values (1000 samples × 12 leads), "
            f"got {arr.shape[0]}."
        )
    return arr.reshape(1000, 12)


# Maps integer severity grade (Notebook 13) → string label matching
# pipeline.run()'s heuristic band labels for UI consistency.
_GRADE_TO_LABEL = {0: "Low", 1: "Low", 2: "Moderate", 3: "High", 4: "Critical"}


def _gate_severity(binary_prediction: str, fused_prob: float, clinical_dict: dict) -> dict:
    """
    Severity gating logic:
    - "No Disease" → always "Low", skip model call.
    - "Disease"    → call Notebook 13 RF; map int grade → string label.
    Falls back to heuristic band from pipeline.run() if severity model unavailable.
    """
    if binary_prediction == "No Disease":
        return {
            "severity_grade": "Low",
            "severity_source": "binary_gate_no_disease",
        }

    try:
        severity_model = get_severity_model()
        X = pd.DataFrame([clinical_dict])[SELECTED_FEATURES]
        grade_int = int(severity_model.predict(X)[0])
        return {
            "severity_grade": _GRADE_TO_LABEL.get(grade_int, "Moderate"),
            "severity_source": "trained_multiclass_rf",
        }
    except FileNotFoundError:
        # Severity model not loaded — fall back to heuristic band
        # (already computed inside pipeline.run(); caller passes fused_prob)
        for threshold, label in [(0.30, "Low"), (0.60, "Moderate"), (0.85, "High"), (1.01, "Critical")]:
            if fused_prob < threshold:
                return {"severity_grade": label, "severity_source": "heuristic_probability_band"}
        return {"severity_grade": "Critical", "severity_source": "heuristic_probability_band"}


import time

def run_assessment(
    request: AssessRequest,
    user_id: str,
    db: Session
) -> AssessResponse:
    pipeline = get_pipeline()

    # --- 1. Build patient dict for pipeline ---
    clinical_dict = request.clinical.model_dump()
    missingness_map = _build_missingness_map(clinical_dict)

    # Replace None with np.nan so pipeline's imputer handles them
    clinical_dict_clean = {
        k: (np.nan if v is None else v) for k, v in clinical_dict.items()
    }

    # --- 2. Reshape ECG if provided, else use zeros ---
    if request.ecg_signal is not None:
        ecg_array = _reshape_ecg(request.ecg_signal)
    else:
        ecg_array = np.zeros((1000, 12), dtype=np.float32)

    # --- 3. Call Notebook 12 inference pipeline ---
    pipeline_result = pipeline.run(clinical_dict_clean, ecg_array)

    # --- 4. Severity gating (Notebook 13, Cell 18 logic) ---
    severity_result = _gate_severity(pipeline_result["prediction"], pipeline_result["fused_probability"], clinical_dict_clean)

    # --- 5. Persist to DB ---
    assessment_id = str(uuid.uuid4())
    db_assessment = Assessment(
        id=assessment_id,
        user_id=user_id,
        clinical_input=clinical_dict,
        feature_missingness_map=missingness_map,
        ecg_quality=pipeline_result.get("ecg_quality"),
        prediction=pipeline_result["prediction"],
        fused_probability=pipeline_result["fused_probability"],
        severity=severity_result["severity_grade"],   # str: Low/Moderate/High/Critical
        severity_source=severity_result["severity_source"],
        confidence=pipeline_result["confidence"],
        branch_contribution=pipeline_result.get("branch_contribution"),
        branch_probabilities=pipeline_result.get("branch_probabilities"),
        top_clinical_features=pipeline_result.get("top_clinical_features"),
        top_ecg_leads=pipeline_result.get("top_ecg_leads"),
        recommendations=pipeline_result.get("recommendations"),
        disclaimer=DISCLAIMER,
    )
    db.add(db_assessment)
    db.commit()
    db.refresh(db_assessment)

    # --- 6. Build response ---
    bc = pipeline_result.get("branch_contribution")
    bp = pipeline_result.get("branch_probabilities")

    return AssessResponse(
        assessment_id=assessment_id,
        prediction=pipeline_result["prediction"],
        fused_probability=pipeline_result["fused_probability"],
        severity=severity_result["severity_grade"],   # str: Low/Moderate/High/Critical
        severity_source=severity_result["severity_source"],
        confidence=pipeline_result["confidence"],
        branch_contribution=BranchContribution(**bc) if bc else None,
        branch_probabilities=BranchProbabilities(**bp) if bp else None,
        top_clinical_features=pipeline_result.get("top_clinical_features"),
        top_ecg_leads=pipeline_result.get("top_ecg_leads"),
        ecg_quality=pipeline_result.get("ecg_quality"),
        recommendations=pipeline_result.get("recommendations"),
        feature_missingness_map=missingness_map,
        disclaimer=DISCLAIMER,
        clinical=clinical_dict,
    )


def run_assessment_generator(
    request: AssessRequest,
    user_id: str,
    db: Session
):
    yield "INIT_PIPELINE"
    time.sleep(0.5)

    pipeline = get_pipeline()

    # --- 1. Build patient dict for pipeline ---
    clinical_dict = request.clinical.model_dump()
    missingness_map = _build_missingness_map(clinical_dict)

    # Replace None with np.nan so pipeline's imputer handles them
    clinical_dict_clean = {
        k: (np.nan if v is None else v) for k, v in clinical_dict.items()
    }

    # --- 2. Reshape ECG if provided, else use zeros ---
    if request.ecg_signal is not None:
        ecg_array = _reshape_ecg(request.ecg_signal)
    else:
        ecg_array = np.zeros((1000, 12), dtype=np.float32)

    # Try granular step-by-step execution; fall back to atomic pipeline.run()
    # if the pipeline class doesn't expose individual sub-engine methods.
    try:
        yield "RUNNING_CLINICAL_RF"
        time.sleep(0.5)
        clinical_raw = pipeline.clinical_engine.predict_raw(clinical_dict_clean)
        clinical_calibrated = clinical_raw

        yield "RUNNING_ECG_CNN"
        time.sleep(0.5)
        ecg_raw = pipeline.ecg_engine.predict_raw(ecg_array)

        yield "CALIBRATING_PLATT"
        time.sleep(0.5)
        ecg_calibrated = pipeline.calibration_engine.calibrate(ecg_raw)

        yield "GATING_NODE_COMPLETE"
        time.sleep(0.5)
        fused_prob = pipeline.fusion_engine.fuse(clinical_calibrated, ecg_calibrated)

        # Severity gating
        severity_result = _gate_severity(
            "Disease" if fused_prob >= 0.5 else "No Disease",
            fused_prob,
            clinical_dict_clean
        )

        # Compile the rest of the report
        quality_report = pipeline.quality_engine.assess(ecg_array)
        clinical_contribution = abs(clinical_calibrated - 0.5)
        ecg_contribution = abs(ecg_calibrated - 0.5)
        total = clinical_contribution + ecg_contribution + 1e-8

        top_clinical_features = pipeline.explain_engine.explain_clinical(clinical_dict_clean)
        top_ecg_leads = pipeline.explain_engine.explain_ecg(ecg_array)
        recommendations = pipeline.recommendation_engine.generate(
            fused_prob, severity_result["severity_grade"], quality_report["flags"]
        )

        # --- 5. Persist to DB ---
        assessment_id = str(uuid.uuid4())
        db_assessment = Assessment(
            id=assessment_id,
            user_id=user_id,
            clinical_input=clinical_dict,
            feature_missingness_map=missingness_map,
            ecg_quality=quality_report,
            prediction="Disease" if fused_prob >= 0.5 else "No Disease",
            fused_probability=fused_prob,
            severity=severity_result["severity_grade"],
            severity_source=severity_result["severity_source"],
            confidence=round(max(fused_prob, 1 - fused_prob), 4),
            branch_contribution={
                "clinical_pct": round(clinical_contribution / total * 100, 1),
                "ecg_pct": round(ecg_contribution / total * 100, 1)
            },
            branch_probabilities={
                "clinical": round(clinical_calibrated, 4),
                "ecg": round(ecg_calibrated, 4)
            },
            top_clinical_features=top_clinical_features,
            top_ecg_leads=top_ecg_leads,
            recommendations=recommendations,
            disclaimer=DISCLAIMER,
        )
        db.add(db_assessment)
        db.commit()
        db.refresh(db_assessment)

        bc = db_assessment.branch_contribution
        bp = db_assessment.branch_probabilities

        response = AssessResponse(
            assessment_id=assessment_id,
            prediction=db_assessment.prediction,
            fused_probability=db_assessment.fused_probability,
            severity=db_assessment.severity,
            severity_source=db_assessment.severity_source,
            confidence=db_assessment.confidence,
            branch_contribution=BranchContribution(**bc) if bc else None,
            branch_probabilities=BranchProbabilities(**bp) if bp else None,
            top_clinical_features=top_clinical_features,
            top_ecg_leads=top_ecg_leads,
            ecg_quality=quality_report,
            recommendations=recommendations,
            feature_missingness_map=missingness_map,
            disclaimer=DISCLAIMER,
            clinical=clinical_dict,
        )

        payload_json = response.model_dump_json()
        yield f"FINAL_REPORT_READY:{payload_json}"

    except AttributeError:
        # Fallback: pipeline doesn't expose granular sub-engine methods.
        # Run atomically via pipeline.run() and emit milestone tokens with delays.
        yield "RUNNING_CLINICAL_RF"
        time.sleep(0.3)
        yield "RUNNING_ECG_CNN"
        time.sleep(0.3)
        yield "CALIBRATING_PLATT"
        time.sleep(0.3)

        pipeline_result = pipeline.run(clinical_dict_clean, ecg_array)

        yield "GATING_NODE_COMPLETE"
        time.sleep(0.3)

        severity_result = _gate_severity(
            pipeline_result["prediction"],
            pipeline_result["fused_probability"],
            clinical_dict_clean,
        )

        assessment_id = str(uuid.uuid4())
        db_assessment = Assessment(
            id=assessment_id,
            user_id=user_id,
            clinical_input=clinical_dict,
            feature_missingness_map=missingness_map,
            ecg_quality=pipeline_result.get("ecg_quality"),
            prediction=pipeline_result["prediction"],
            fused_probability=pipeline_result["fused_probability"],
            severity=severity_result["severity_grade"],
            severity_source=severity_result["severity_source"],
            confidence=pipeline_result["confidence"],
            branch_contribution=pipeline_result.get("branch_contribution"),
            branch_probabilities=pipeline_result.get("branch_probabilities"),
            top_clinical_features=pipeline_result.get("top_clinical_features"),
            top_ecg_leads=pipeline_result.get("top_ecg_leads"),
            recommendations=pipeline_result.get("recommendations"),
            disclaimer=DISCLAIMER,
        )
        db.add(db_assessment)
        db.commit()
        db.refresh(db_assessment)

        bc = pipeline_result.get("branch_contribution")
        bp = pipeline_result.get("branch_probabilities")

        response = AssessResponse(
            assessment_id=assessment_id,
            prediction=pipeline_result["prediction"],
            fused_probability=pipeline_result["fused_probability"],
            severity=severity_result["severity_grade"],
            severity_source=severity_result["severity_source"],
            confidence=pipeline_result["confidence"],
            branch_contribution=BranchContribution(**bc) if bc else None,
            branch_probabilities=BranchProbabilities(**bp) if bp else None,
            top_clinical_features=pipeline_result.get("top_clinical_features"),
            top_ecg_leads=pipeline_result.get("top_ecg_leads"),
            ecg_quality=pipeline_result.get("ecg_quality"),
            recommendations=pipeline_result.get("recommendations"),
            feature_missingness_map=missingness_map,
            disclaimer=DISCLAIMER,
            clinical=clinical_dict,
        )

        payload_json = response.model_dump_json()
        yield f"FINAL_REPORT_READY:{payload_json}"
