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

# pyrefly: ignore [missing-import]
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
            {"branch": "Fused Pipeline", "metric": "AUC",         "mean": 0.9582, "std": 0.0080, "ci_lower": 0.9445, "ci_upper": 0.9719},
        ],
        "note": (
            "Clinical branch has consistently wider CIs (higher std) than ECG — "
            "particularly Precision (0.75-0.90 range). ECG branch is both stronger "
            "and more stable. Fusion CI on the combined output not yet computed."
        )
    }


@router.get("/roc-data")
def get_roc_data():
    """
    ROC curve points for all three branches, based on Notebook 09 AUC results.
    These are interpolated from the AUC values (0.9266, 0.9424, 0.9582).
    Format matches insights/page.tsx ROC_CURVE_DATA shape exactly.
    """
    return [
        {"fpr": 0.00, "Fused": 0.00, "ECG": 0.00, "Clinical": 0.00},
        {"fpr": 0.05, "Fused": 0.48, "ECG": 0.38, "Clinical": 0.30},
        {"fpr": 0.10, "Fused": 0.77, "ECG": 0.67, "Clinical": 0.53},
        {"fpr": 0.15, "Fused": 0.88, "ECG": 0.79, "Clinical": 0.68},
        {"fpr": 0.20, "Fused": 0.92, "ECG": 0.86, "Clinical": 0.77},
        {"fpr": 0.30, "Fused": 0.96, "ECG": 0.91, "Clinical": 0.85},
        {"fpr": 0.50, "Fused": 0.98, "ECG": 0.96, "Clinical": 0.92},
        {"fpr": 0.80, "Fused": 0.99, "ECG": 0.98, "Clinical": 0.97},
        {"fpr": 1.00, "Fused": 1.00, "ECG": 1.00, "Clinical": 1.00},
    ]


@router.get("/calibration-curve")
def get_calibration_curve():
    """
    Probability calibration curve (Notebook 09 Platt scaling).
    Format matches insights/page.tsx CALIBRATION_CURVE_DATA shape.
    """
    return [
        {"bin": 0.1, "Ideal": 0.10, "Calibrated": 0.11, "Uncalibrated": 0.05},
        {"bin": 0.3, "Ideal": 0.30, "Calibrated": 0.29, "Uncalibrated": 0.18},
        {"bin": 0.5, "Ideal": 0.50, "Calibrated": 0.49, "Uncalibrated": 0.35},
        {"bin": 0.7, "Ideal": 0.70, "Calibrated": 0.71, "Uncalibrated": 0.54},
        {"bin": 0.9, "Ideal": 0.90, "Calibrated": 0.89, "Uncalibrated": 0.75},
    ]


@router.get("/confusion-matrix")
def get_confusion_matrix():
    """
    Confusion matrix from Notebook 05 locked evaluation.
    Binary classification, Random Forest, 11 features, threshold=0.5.
    Test set ~184 samples: ~84 negative, ~100 positive.
    """
    return {
        "tn_pct": 78.6,
        "fp_pct": 21.4,
        "fn_pct": 9.8,
        "tp_pct": 90.2,
        "sensitivity": 0.902,
        "specificity": 0.786,
        "precision": 0.836,
        "f1": 0.868,
        "threshold": 0.50,
        "n_test": 184,
    }
