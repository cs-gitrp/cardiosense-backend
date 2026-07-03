"""
Model Insights endpoints — powers the Insights dashboard in process.md Section 7.

Three endpoints:
  GET /insights/calibration     — Notebook 09 Brier/ECE/AUC numbers (per-branch)
  GET /insights/model-comparison — Notebook 03/05 ablation table (17 vs 11 features)
  GET /insights/bootstrap-ci    — Notebook 11 bootstrap CI table

These return hardcoded notebook results for now. Once ModelRun rows are
populated (by running a small seed script after deployment), swap the
hardcoded dicts for:
    db.query(ModelRun).filter(ModelRun.is_active == True).all()
"""

from fastapi import APIRouter

router = APIRouter(prefix="/insights", tags=["insights"])


@router.get("/calibration")
def get_calibration_metrics():
    """
    Notebook 09 calibration results.
    ECG branch is stronger AND better-calibrated on all three metrics.
    """
    return {
        "clinical": {
            "auc": 0.9262,
            "brier_score": 0.1130,
            "expected_calibration_error": 0.0653,
        },
        "ecg": {
            "auc": 0.9423,
            "brier_score": 0.0943,
            "expected_calibration_error": 0.0340,
        },
        "note": (
            "Calibration applied via Platt scaling (CalibratedClassifierCV). "
            "ECG branch dominates on all three metrics — AUC, Brier, and ECE. "
            "Fusion uses confidence-adaptive weighting (gamma=3) to exploit this asymmetry."
        )
    }


@router.get("/model-comparison")
def get_model_comparison():
    """
    Notebook 05 feature-selection ablation:
    17-feature baseline vs 11 consensus-selected features.
    All 4 models passed the pre-committed F1 drop < 2% rule.
    """
    return {
        "rule": "F1 drop < 2% for feature reduction to be accepted",
        "results": [
            {"model": "Logistic Regression", "delta_accuracy": +2.18, "delta_f1": +2.44, "verdict": "Improved"},
            {"model": "Random Forest",       "delta_accuracy": +1.08, "delta_f1": +1.35, "verdict": "Improved"},
            {"model": "SVM",                 "delta_accuracy": -1.09, "delta_f1": -0.46, "verdict": "Passed"},
            {"model": "XGBoost",             "delta_accuracy":  0.00, "delta_f1": -0.15, "verdict": "Passed"},
        ],
        "locked_best": {
            "model": "Random Forest",
            "features": 11,
            "accuracy": 0.8478,
            "f1": 0.8679,
            "recall": 0.902,
        }
    }


@router.get("/bootstrap-ci")
def get_bootstrap_ci():
    """
    Notebook 11 bootstrap confidence intervals (n=1000, threshold=0.50, SEED=42).
    Per-branch (clinical / ECG) — fusion CI not yet computed (see data_notes.md caveat).
    """
    return {
        "n_bootstrap": 1000,
        "results": [
            {"branch": "Clinical", "metric": "Accuracy",          "mean": 0.8364, "std": 0.0276, "ci_lower": 0.7826, "ci_upper": 0.8913},
            {"branch": "ECG",      "metric": "Accuracy",          "mean": 0.8729, "std": 0.0062, "ci_lower": 0.8606, "ci_upper": 0.8853},
            {"branch": "Clinical", "metric": "Precision",         "mean": 0.8268, "std": 0.0366, "ci_lower": 0.7523, "ci_upper": 0.8957},
            {"branch": "ECG",      "metric": "Precision",         "mean": 0.9003, "std": 0.0069, "ci_lower": 0.8874, "ci_upper": 0.9135},
            {"branch": "Clinical", "metric": "Recall",            "mean": 0.8920, "std": 0.0317, "ci_lower": 0.8230, "ci_upper": 0.9469},
            {"branch": "ECG",      "metric": "Recall",            "mean": 0.8756, "std": 0.0082, "ci_lower": 0.8594, "ci_upper": 0.8916},
            {"branch": "Clinical", "metric": "F1 Score",          "mean": 0.8576, "std": 0.0258, "ci_lower": 0.8039, "ci_upper": 0.9065},
            {"branch": "ECG",      "metric": "F1 Score",          "mean": 0.8877, "std": 0.0058, "ci_lower": 0.8764, "ci_upper": 0.8988},
            {"branch": "Clinical", "metric": "AUC",               "mean": 0.9266, "std": 0.0178, "ci_lower": 0.8887, "ci_upper": 0.9582},
            {"branch": "ECG",      "metric": "AUC",               "mean": 0.9424, "std": 0.0040, "ci_lower": 0.9347, "ci_upper": 0.9504},
            {"branch": "Clinical", "metric": "Average Precision", "mean": 0.9427, "std": 0.0157, "ci_lower": 0.9105, "ci_upper": 0.9705},
            {"branch": "ECG",      "metric": "Average Precision", "mean": 0.9627, "std": 0.0029, "ci_lower": 0.9568, "ci_upper": 0.9681},
        ],
        "note": (
            "Clinical branch has consistently wider CIs (higher std) than ECG — "
            "particularly Precision (0.75-0.90 range). ECG branch is both stronger "
            "and more stable. Fusion CI on the combined output not yet computed."
        )
    }
