"""
python -m pip install "scikit-learn==1.6.1"
python diagnose_pipeline.py
"""
import os, sys, pickle, warnings, importlib.util
warnings.filterwarnings("ignore")

MODEL_DIR    = os.environ.get("MODEL_DIR", "./ml_artifacts")
PIPELINE_DIR = os.path.join(MODEL_DIR, "cardiosense_pipeline")
sys.path.insert(0, PIPELINE_DIR)
SEP = "=" * 60

# ── 1. File inventory ─────────────────────────────────────────
print(SEP); print("STEP 1 — File inventory"); print(SEP)
for fname in ["cardiosense_pipeline.pkl","ecg_model.keras","clinical_rf.pkl",
              "04_selected_features.csv","manifest.json","pipeline_class.py"]:
    fp = os.path.join(PIPELINE_DIR, fname)
    e  = os.path.exists(fp)
    print(f"  {'OK     ' if e else 'MISSING'} {fname}" +
          (f"  ({os.path.getsize(fp):,} bytes)" if e else ""))

# ── 2. Bootstrap: import pipeline_class, inject into __main__ ─
print(); print(SEP); print("STEP 2 — Bootstrap pipeline_class.py → __main__"); print(SEP)
try:
    spec = importlib.util.spec_from_file_location(
        "pipeline_class", os.path.join(PIPELINE_DIR, "pipeline_class.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pipeline_class"] = mod
    spec.loader.exec_module(mod)
    import __main__
    injected = []
    for name in dir(mod):
        if not name.startswith("_"):
            setattr(__main__, name, getattr(mod, name))
            injected.append(name)
    print("OK — injected:", injected)
    # Verify key names are present
    for required in ["adaptive_fusion", "confidence", "FusionEngine",
                     "ClinicalEngine", "CardioSensePipeline"]:
        status = "OK" if hasattr(__main__, required) else "MISSING"
        print(f"  {status}  __main__.{required}")
except Exception as e:
    print(f"FAILED — {type(e).__name__}: {e}"); sys.exit(1)

# ── 3. Load component pkl ──────────────────────────────────────
print(); print(SEP); print("STEP 3 — Load cardiosense_pipeline.pkl"); print(SEP)
try:
    with open(os.path.join(PIPELINE_DIR, "cardiosense_pipeline.pkl"), "rb") as f:
        components = pickle.load(f)
    print(f"OK — keys: {list(components.keys())}")
    print(f"  clinical_features: {components.get('clinical_features')}")
except Exception as e:
    print(f"FAILED — {type(e).__name__}: {e}"); sys.exit(1)

# ── 4. Load ecg_model.keras ────────────────────────────────────
print(); print(SEP); print("STEP 4 — Load ecg_model.keras"); print(SEP)
try:
    import tensorflow as tf
    cnn = tf.keras.models.load_model(os.path.join(PIPELINE_DIR, "ecg_model.keras"))
    print(f"OK — input: {cnn.input_shape}  output: {cnn.output_shape}")
except Exception as e:
    print(f"FAILED — {type(e).__name__}: {e}")

# ── 5. Load clinical_rf.pkl ────────────────────────────────────
print(); print(SEP); print("STEP 5 — Load clinical_rf.pkl"); print(SEP)
try:
    import joblib
    rf = joblib.load(os.path.join(PIPELINE_DIR, "clinical_rf.pkl"))
    print(f"OK — {type(rf).__name__}")
except Exception as e:
    print(f"FAILED — {type(e).__name__}: {e}")

# ── 6. Severity model ──────────────────────────────────────────
print(); print(SEP); print("STEP 6 — Load severity model"); print(SEP)
try:
    import joblib
    sev = joblib.load(os.path.join(MODEL_DIR, "severity", "13_severity_rf.pkl"))
    print(f"OK — classes: {list(sev.classes_)}")
except Exception as e:
    print(f"FAILED — {type(e).__name__}: {e}")

# ── 7. Quick pipeline smoke test ──────────────────────────────
print(); print(SEP); print("STEP 7 — Smoke test: build + run pipeline"); print(SEP)
try:
    import numpy as np
    ecg_engine     = mod.ECGEngine(cnn)
    explain_engine = mod.ExplainabilityEngine(rf, cnn, components["clinical_features"])
    pipeline = mod.CardioSensePipeline(
        clinical_engine=components["clinical_engine"],
        ecg_engine=ecg_engine,
        quality_engine=components["quality_engine"],
        calibration_engine=components["ecg_calibration_engine"],
        fusion_engine=components["fusion_engine"],
        explain_engine=explain_engine,
        recommendation_engine=components["recommendation_engine"],
    )
    # Dummy inputs
    dummy_patient = {f: 0.5 for f in components["clinical_features"]}
    dummy_ecg     = np.random.randn(1000, 12).astype(np.float32)
    result = pipeline.run(dummy_patient, dummy_ecg)
    print("OK — pipeline.run() returned:")
    for k, v in result.items():
        print(f"  {k}: {v}")
except Exception as e:
    import traceback
    print(f"FAILED — {type(e).__name__}: {e}")
    traceback.print_exc()

print(); print(SEP); print("Done. Paste output into chat."); print(SEP)
