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
import time

from sqlalchemy.orm import Session

from app.schemas.assessment import AssessRequest, AssessResponse, BranchContribution, BranchProbabilities
from app.models.assessment import Assessment
from app.services.model_loader import get_pipeline, get_severity_model

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
    return {feat: (clinical_dict.get(feat) is None) for feat in SELECTED_FEATURES}


def _reshape_ecg(ecg_flat: list[float]) -> np.ndarray:
    arr = np.array(ecg_flat, dtype=np.float32)
    if arr.shape != (12000,):
        raise ValueError(f"ecg_signal must have exactly 12000 values, got {arr.shape[0]}.")
    return arr.reshape(1000, 12)


_GRADE_TO_LABEL = {0: "Low", 1: "Low", 2: "Moderate", 3: "High", 4: "Critical"}


def _gate_severity(binary_prediction: str, fused_prob: float, clinical_dict: dict) -> dict:
    """
    Severity gating function linked to Notebook 13.
    Ensures active disease classifications step out of the resting 'Low' baseline to keep indicators clear.
    """
    if binary_prediction == "No Disease":
        return {"severity_grade": "Low", "severity_source": "binary_gate_no_disease"}

    try:
        severity_model = get_severity_model()
        X = pd.DataFrame([clinical_dict])[SELECTED_FEATURES]
        grade_int = int(severity_model.predict(X)[0])
        computed_label = _GRADE_TO_LABEL.get(grade_int, "Moderate")
        # Enforce minimum 'Moderate' severity for positive disease predictions to resolve visual contradictions
        final_label = "Moderate" if computed_label == "Low" else computed_label
        return {"severity_grade": final_label, "severity_source": "trained_multiclass_rf"}
    except:
        for threshold, label in [(0.30, "Low"), (0.60, "Moderate"), (0.85, "High"), (1.01, "Critical")]:
            if fused_prob < threshold:
                return {"severity_grade": "Moderate" if label == "Low" else label, "severity_source": "heuristic_probability_band"}
        return {"severity_grade": "Critical", "severity_source": "heuristic_probability_band"}


def _compile_prompt_summaries(fused_prob: float, prediction: str, severity: str, source: str, clinical_dict: dict, top_leads: list | None) -> dict:
    """Generates data-driven clinical summaries straight from model parameters (Prompt 1)."""
    cp_val = clinical_dict.get("cp", 0)
    oldpeak_val = clinical_dict.get("oldpeak", 0.0)
    prob_pct = round(fused_prob * 100)
    
    # 1. Clinician Oriented Summary
    doc_summary = (
        f"The submitted clinical profile (Chest Pain Type {cp_val}, Oldpeak {oldpeak_val} mm) "
        f"together with the ECG branch produced a fused probability of {prob_pct}%. "
        f"Severity classification: {severity or 'Moderate'}, evaluated via the {source} framework. "
        f"This represents a screening result only."
    )
    
    # 2. Patient Oriented Explanation
    if prediction == "Disease":
        pat_summary = (
            f"The AI model identified distinct clinical markers across your vital observations that suggest an "
            f"elevated cardiac risk configuration categorized at a {severity.lower()} severity profile. This does not confirm "
            f"active disease, but it suggests discussing this screening log with a healthcare professional."
        )
    else:
        pat_summary = (
            "Your screening profile indicates low baseline risk parameters. The neural fusion network determined "
            "that your vitals and wave indicators fall comfortably within reference boundaries. Maintain standard health tracking."
        )
        
    # 3. Strongest ECG Lead Annotation
    if top_leads and len(top_leads) > 0:
        strongest = top_leads[0]
        lead_name = strongest.get("lead") if isinstance(strongest, dict) else getattr(strongest, "lead", "V5")
        attr_score = strongest.get("attribution") if isinstance(strongest, dict) else getattr(strongest, "attribution", 0.0)
        lead_annotation = f"The strongest ECG attribution occurred in Lead {lead_name} (+{attr_score:.3f}), indicating that this lead contributed most strongly to the neural decision."
    else:
        lead_annotation = "No ECG recording was available."

    return {
        "doctor_summary": doc_summary,
        "patient_summary": pat_summary,
        "lead_annotation": lead_annotation,
        "display_waveforms": None  # Bounded placeholder strategy (Prompt 1 & 3)
    }


def run_assessment(
    request: AssessRequest,
    user_id: str,
    db: Session
) -> AssessResponse:
    pipeline = get_pipeline()
    clinical_dict = request.clinical.model_dump()
    missingness_map = _build_missingness_map(clinical_dict)
    clinical_dict_clean = {k: (np.nan if v is None else v) for k, v in clinical_dict.items()}

    if request.ecg_signal is not None:
        ecg_array = _reshape_ecg(request.ecg_signal)
    else:
        ecg_array = np.zeros((1000, 12), dtype=np.float32)

    pipeline_result = pipeline.run(clinical_dict_clean, ecg_array)

    if request.ecg_signal is None:
        pipeline_result["branch_contribution"] = {"clinical_pct": 100.0, "ecg_pct": 0.0}
        current_bp = pipeline_result.get("branch_probabilities") or {}
        pipeline_result["branch_probabilities"] = {
            "clinical": current_bp.get("clinical", pipeline_result["fused_probability"]),
            "ecg": None
        }
        pipeline_result["ecg_quality"] = {"quality_score": 0, "flags": ["No ECG recording provided"], "is_acceptable": False}

    severity_result = _gate_severity(pipeline_result["prediction"], pipeline_result["fused_probability"], clinical_dict_clean)

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

    enriched_content = _compile_prompt_summaries(
        pipeline_result["fused_probability"], pipeline_result["prediction"],
        severity_result["severity_grade"], severity_result["severity_source"],
        clinical_dict, pipeline_result.get("top_ecg_leads")
    )

    return AssessResponse(
        assessment_id=assessment_id,
        prediction=pipeline_result["prediction"],
        fused_probability=pipeline_result["fused_probability"],
        severity=severity_result["severity_grade"],
        severity_source=severity_result["severity_source"],
        confidence=pipeline_result["confidence"],
        branch_contribution=BranchContribution(**pipeline_result["branch_contribution"]) if pipeline_result.get("branch_contribution") else None,
        branch_probabilities=BranchProbabilities(**pipeline_result["branch_probabilities"]) if pipeline_result.get("branch_probabilities") else None,
        top_clinical_features=pipeline_result.get("top_clinical_features"),
        top_ecg_leads=pipeline_result.get("top_ecg_leads"),
        ecg_quality=pipeline_result.get("ecg_quality"),
        recommendations=pipeline_result.get("recommendations"),
        feature_missingness_map=missingness_map,
        disclaimer=DISCLAIMER,
        clinical=clinical_dict,
        **enriched_content
    )


def run_assessment_generator(
    request: AssessRequest,
    user_id: str,
    db: Session
):
    yield "INIT_PIPELINE"
    time.sleep(0.1)

    pipeline = get_pipeline()
    clinical_dict = request.clinical.model_dump()
    missingness_map = _build_missingness_map(clinical_dict)
    clinical_dict_clean = {k: (np.nan if v is None else v) for k, v in clinical_dict.items()}

    if request.ecg_signal is not None:
        ecg_array = _reshape_ecg(request.ecg_signal)
    else:
        rose_zeros = np.zeros((1000, 12), dtype=np.float32)
        ecg_array = rose_zeros

    try:
        yield "RUNNING_CLINICAL_RF"
        time.sleep(0.1)
        clinical_raw = pipeline.clinical_engine.predict_raw(clinical_dict_clean)

        yield "RUNNING_ECG_CNN"
        time.sleep(0.1)
        ecg_raw = pipeline.ecg_engine.predict_raw(ecg_array)

        yield "CALIBRATING_PLATT"
        time.sleep(0.1)
        ecg_calibrated = pipeline.calibration_engine.calibrate(ecg_raw)

        yield "GATING_NODE_COMPLETE"
        time.sleep(0.1)
        fused_prob = pipeline.fusion_engine.fuse(clinical_raw, ecg_calibrated)

        severity_result = _gate_severity("Disease" if fused_prob >= 0.5 else "No Disease", fused_prob, clinical_dict_clean)

        if request.ecg_signal is None:
            quality_report = {"quality_score": 0, "flags": ["No ECG recording provided"], "is_acceptable": False}
            branch_contrib = {"clinical_pct": 100.0, "ecg_pct": 0.0}
            branch_probs = {"clinical": round(float(clinical_raw), 4), "ecg": None}
            top_ecg_leads = []
        else:
            quality_report = pipeline.quality_engine.assess(ecg_array)
            total = abs(clinical_raw - 0.5) + abs(ecg_calibrated - 0.5) + 1e-8
            branch_contrib = {"clinical_pct": round(abs(clinical_raw - 0.5) / total * 100, 1), "ecg_pct": round(abs(ecg_calibrated - 0.5) / total * 100, 1)}
            branch_probs = {"clinical": round(float(clinical_raw), 4), "ecg": round(float(ecg_calibrated), 4)}
            top_ecg_leads = pipeline.explain_engine.explain_ecg(ecg_array)

        top_clinical_features = pipeline.explain_engine.explain_clinical(clinical_dict_clean)
        recommendations = pipeline.recommendation_engine.generate(fused_prob, severity_result["severity_grade"], quality_report["flags"])

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
            branch_contribution=branch_contrib,
            branch_probabilities=branch_probs,
            top_clinical_features=top_clinical_features,
            top_ecg_leads=top_ecg_leads,
            recommendations=recommendations,
            disclaimer=DISCLAIMER,
        )
        db.add(db_assessment)
        db.commit()

        enriched_content = _compile_prompt_summaries(
            fused_prob, db_assessment.prediction,
            severity_result["severity_grade"], severity_result["severity_source"],
            clinical_dict, top_ecg_leads
        )

        response = AssessResponse(
            assessment_id=assessment_id,
            prediction=db_assessment.prediction,
            fused_probability=db_assessment.fused_probability,
            severity=db_assessment.severity,
            severity_source=db_assessment.severity_source,
            confidence=db_assessment.confidence,
            branch_contribution=BranchContribution(**branch_contrib),
            branch_probabilities=BranchProbabilities(**branch_probs),
            top_clinical_features=top_clinical_features,
            top_ecg_leads=top_ecg_leads,
            ecg_quality=quality_report,
            recommendations=recommendations,
            feature_missingness_map=missingness_map,
            disclaimer=DISCLAIMER,
            clinical=clinical_dict,
            **enriched_content
        )

        yield f"FINAL_REPORT_READY:{response.model_dump_json()}"

    except Exception as e:
        yield f"ERROR:{str(e)}"