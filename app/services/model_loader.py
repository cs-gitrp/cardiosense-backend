"""
Loads and caches the CardioSense inference pipeline.

Loading order (matters — pickle resolves names at load time):
  1. pipeline_class.py  → class defs + fusion fns injected into __main__
  2. cardiosense_pipeline.pkl (dict) → component instances
  3. ecg_model.keras    → Keras CNN
  4. clinical_rf.pkl    → RF for SHAP
  5. Assemble CardioSensePipeline

NOTE: 09_confidence_adaptive_fusion.pkl (43 bytes) is intentionally
skipped. It serialized adaptive_fusion by reference, not by value,
making it unloadable without the function already in scope (circular).
The function is defined directly in pipeline_class.py instead.
"""

import os, sys, pickle, joblib, importlib.util
from app.core.config import settings

_pipeline       = None
_severity_model = None


def _pipeline_dir() -> str:
    p = os.path.join(settings.MODEL_DIR, "cardiosense_pipeline")
    if not os.path.isdir(p):
        raise FileNotFoundError(f"Pipeline directory not found: {p}")
    return p


def _bootstrap(pipeline_dir: str):
    """
    Imports pipeline_class.py and injects all names into __main__
    so pickle can resolve every class + function reference when
    deserializing cardiosense_pipeline.pkl.
    """
    if pipeline_dir not in sys.path:
        sys.path.insert(0, pipeline_dir)

    class_file = os.path.join(pipeline_dir, "pipeline_class.py")
    if not os.path.exists(class_file):
        raise FileNotFoundError(
            f"pipeline_class.py not found at {class_file}. "
            "It should be inside ml_artifacts/cardiosense_pipeline/."
        )

    spec = importlib.util.spec_from_file_location("pipeline_class", class_file)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules["pipeline_class"] = mod
    spec.loader.exec_module(mod)

    import __main__
    # Inject everything from the module into __main__ so pickle finds it
    for name in dir(mod):
        if not name.startswith("_"):
            setattr(__main__, name, getattr(mod, name))

    return mod


def get_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    pipeline_dir = _pipeline_dir()
    mod = _bootstrap(pipeline_dir)

    # ── Load component dict ──────────────────────────────────
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # suppress sklearn version warnings
        with open(os.path.join(pipeline_dir, "cardiosense_pipeline.pkl"), "rb") as f:
            components = pickle.load(f)

    clinical_engine        = components["clinical_engine"]
    ecg_calibration_engine = components["ecg_calibration_engine"]
    fusion_engine          = components["fusion_engine"]
    quality_engine         = components["quality_engine"]
    recommendation_engine  = components["recommendation_engine"]
    clinical_features      = components["clinical_features"]

    # ── Load ECG CNN ──────────────────────────────────────────
    import tensorflow as tf
    cnn_model = tf.keras.models.load_model(
        os.path.join(pipeline_dir, "ecg_model.keras")
    )

    # ── Load RF for SHAP ──────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rf_model = joblib.load(os.path.join(pipeline_dir, "clinical_rf.pkl"))

    # ── Build missing engines ─────────────────────────────────
    ecg_engine     = mod.ECGEngine(cnn_model)
    explain_engine = mod.ExplainabilityEngine(rf_model, cnn_model, clinical_features)

    # ── Assemble ──────────────────────────────────────────────
    _pipeline = mod.CardioSensePipeline(
        clinical_engine=clinical_engine,
        ecg_engine=ecg_engine,
        quality_engine=quality_engine,
        calibration_engine=ecg_calibration_engine,
        fusion_engine=fusion_engine,
        explain_engine=explain_engine,
        recommendation_engine=recommendation_engine,
    )
    print(f"CardioSensePipeline ready. Features: {clinical_features}")
    return _pipeline


def get_severity_model():
    global _severity_model
    if _severity_model is not None:
        return _severity_model

    severity_dir = os.path.join(settings.MODEL_DIR, "severity")
    for fname in ("13_severity_rf.pkl", "13_severity_xgb.pkl"):
        path = os.path.join(severity_dir, fname)
        if os.path.exists(path):
            _severity_model = joblib.load(path)
            print(f"Severity model loaded: {fname}")
            return _severity_model

    raise FileNotFoundError(f"No severity model found in {severity_dir}.")
