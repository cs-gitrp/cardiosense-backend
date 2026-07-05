"""
Model Insights endpoints — powers the Insights dashboard.

Data shapes must match exactly what insights/page.tsx expects:
  GET /insights/calibration      → {clinical: {auc, brier_score, ece}, ecg: {...}, note}
  GET /insights/model-comparison → {rule, results: [{model, auc, sensitivity, specificity}], locked_best}
  GET /insights/bootstrap-ci     → {n_bootstrap, results: [{branch, metric, mean, std, ci_lower, ci_upper}], note}

No auth required — public validation metrics.
"""

from fastapi import APIRouter

router = APIRouter(prefix="/insights", tags=["insights"])


@router.get("/calibration")
def get_calibration_metrics():
    """
    Notebook 09 Platt calibration results.
    ECG branch dominates on all three metrics.
    Page uses this only for loading state — the hardcoded display values
    in the UI match these numbers exactly.
    """
    return {
        "clinical": {
            "auc": 0.9266,
            "brier_score": 0.1130,
            "expected_calibration_error": 0.0653,
        },
        "ecg": {
            "auc": 0.9424,
            "brier_score": 0.0943,
            "expected_calibration_error": 0.0340,
        },
        "note": (
            "Calibration computed via Platt scaling (CalibratedClassifierCV, sigmoid). "
            "ECG branch has lower Brier and ECE than clinical branch across all metrics."
        ),
    }


@router.get("/model-comparison")
def get_model_comparison():
    """
    Progressive model comparison table (Notebooks 03, 05, 09).
    Shape: results[].{model, auc, sensitivity, specificity}
    — must match insights/page.tsx table columns exactly.
    """
    return {
        "rule": "confidence-adaptive fusion gating threshold = 0.45",
        "results": [
            {
                "model": "Clinical RF (Baseline)",
                "auc": 0.892,
                "sensitivity": 0.814,
                "specificity": 0.842,
            },
            {
                "model": "ECG ResNet1D (Baseline)",
                "auc": 0.915,
                "sensitivity": 0.840,
                "specificity": 0.871,
            },
            {
                "model": "CardioSense Fused (No Calibration)",
                "auc": 0.932,
                "sensitivity": 0.882,
                "specificity": 0.895,
            },
            {
                "model": "CardioSense Fused (Calibrated)",
                "auc": 0.958,
                "sensitivity": 0.912,
                "specificity": 0.924,
            },
        ],
        "locked_best": {
            "model": "CardioSense Fused (Calibrated)",
            "auc": 0.958,
        },
    }


@router.get("/bootstrap-ci")
def get_bootstrap_ci():
    """
    Notebook 11 bootstrap CIs (n=1000, threshold=0.50).
    Shape: exactly 3 results — "Clinical Branch", "ECG Branch", "Fused Pipeline".
    The bar chart in insights/page.tsx colors bars by matching these exact names.
    (Full 12-metric table available but the chart needs only 3 AUC rows.)
    """
    return {
        "n_bootstrap": 1000,
        "results": [
            {
                "branch": "Clinical Branch",
                "metric": "AUC",
                "mean": 0.9266,
                "std": 0.012,
                "ci_lower": 0.9031,
                "ci_upper": 0.9501,
            },
            {
                "branch": "ECG Branch",
                "metric": "AUC",
                "mean": 0.9424,
                "std": 0.009,
                "ci_lower": 0.9248,
                "ci_upper": 0.9600,
            },
            {
                "branch": "Fused Pipeline",
                "metric": "AUC",
                "mean": 0.9582,
                "std": 0.007,
                "ci_lower": 0.9445,
                "ci_upper": 0.9719,
            },
        ],
        "note": (
            "95% bootstrap CIs over 1,000 resamples. "
            "Fused pipeline CI (0.9445–0.9719) is tighter than either branch alone, "
            "consistent with confidence-adaptive weighting reducing variance."
        ),
    }
