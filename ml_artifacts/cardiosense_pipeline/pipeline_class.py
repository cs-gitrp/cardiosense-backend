
# ==========================================================
# Fusion functions — defined here (not loaded from pkl)
# because 09_confidence_adaptive_fusion.pkl serializes the
# function by reference, which requires them to already exist
# in __main__. Defining them here breaks the circular dependency.
# Source: Notebook 09 (gamma=3 variant is the production version).
# ==========================================================

import numpy as _np


def confidence(probability):
    """Converts probability to confidence. Range: 0.5 (uncertain) → 1.0 (certain)."""
    probability = _np.asarray(probability)
    return _np.maximum(probability, 1 - probability)


def adaptive_fusion(clinical_prob, ecg_prob):
    """Linear confidence-weighted fusion (gamma=1 baseline)."""
    clinical_conf = confidence(clinical_prob)
    ecg_conf      = confidence(ecg_prob)
    weight_clinical = clinical_conf / (clinical_conf + ecg_conf)
    weight_ecg      = ecg_conf      / (clinical_conf + ecg_conf)
    fused_probability = weight_clinical * clinical_prob + weight_ecg * ecg_prob
    return {
        "clinical_probability": clinical_prob,
        "ecg_probability":      ecg_prob,
        "clinical_confidence":  clinical_conf,
        "ecg_confidence":       ecg_conf,
        "clinical_weight":      weight_clinical,
        "ecg_weight":           weight_ecg,
        "fused_probability":    fused_probability,
    }


def adaptive_fusion_gamma(clinical_prob, ecg_prob, gamma=3):
    """Gamma-sharpened fusion (gamma=3, production version from Notebook 09)."""
    clinical_conf = confidence(clinical_prob)
    ecg_conf      = confidence(ecg_prob)
    cw = clinical_conf ** gamma
    ew = ecg_conf      ** gamma
    clinical_weight = cw / (cw + ew)
    ecg_weight      = 1 - clinical_weight
    fused_probability = clinical_weight * clinical_prob + ecg_weight * ecg_prob
    return {
        "clinical_weight":   clinical_weight,
        "ecg_weight":        ecg_weight,
        "fused_probability": fused_probability,
    }

"""
pipeline_class.py
-----------------
All class definitions from Notebook 12 (cells 1–8), extracted verbatim.
This file must live at ml_artifacts/cardiosense_pipeline/pipeline_class.py
so that joblib can resolve the class references when deserializing
cardiosense_pipeline.pkl.

DO NOT instantiate anything here — this is a definitions-only module.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import shap


class ECGQualityEngine:
    """Heuristic signal-quality checks. Runs before any ECG prediction."""

    def __init__(self, flatline_std_threshold=1e-3, clip_margin=0.001):
        self.flatline_std_threshold = flatline_std_threshold
        self.clip_margin = clip_margin

    def assess(self, signal):
        # signal shape: (timesteps, 12)
        flags = []
        lead_scores = []

        for lead_idx in range(signal.shape[1]):
            lead = signal[:, lead_idx]
            std = np.std(lead)
            lead_flatline = std < self.flatline_std_threshold
            lead_min, lead_max = lead.min(), lead.max()
            saturated = np.mean((lead <= lead_min + self.clip_margin) |
                                 (lead >= lead_max - self.clip_margin)) > 0.3

            if lead_flatline:
                flags.append(f"Lead {lead_idx+1}: flatline detected")
            if saturated:
                flags.append(f"Lead {lead_idx+1}: possible clipping/saturation")

            lead_scores.append(0 if lead_flatline else (50 if saturated else 100))

        # crude noise estimate: ratio of high-frequency energy (diff-based) to total energy
        diffs = np.diff(signal, axis=0)
        noise_ratio = np.mean(np.var(diffs, axis=0)) / (np.mean(np.var(signal, axis=0)) + 1e-8)
        if noise_ratio > 2.0:
            flags.append("High-frequency noise detected")

        quality_score = float(np.mean(lead_scores))
        if noise_ratio > 2.0:
            quality_score = max(0, quality_score - 20)

        return {
            "quality_score": round(quality_score, 1),
            "flags": flags,
            "is_acceptable": quality_score >= 60 and len(flags) <= 2
        }


class ClinicalEngine:
    def __init__(self, model, feature_order):
        self.model = model
        self.feature_order = feature_order

    def validate_input(self, patient_dict):
        missing = [f for f in self.feature_order if f not in patient_dict]
        if missing:
            raise ValueError(f"Missing clinical features: {missing}")
        return pd.DataFrame([patient_dict])[self.feature_order]

    def predict_raw(self, patient_dict):
        X = self.validate_input(patient_dict)
        raw_prob = self.model.predict_proba(X)[0, 1]
        return float(raw_prob)


class ECGEngine:
    def __init__(self, model, expected_shape=(1000, 12)):
        self.model = model
        self.expected_shape = expected_shape

    def validate_input(self, signal):
        signal = np.asarray(signal)
        if signal.shape != self.expected_shape:
            raise ValueError(f"Expected ECG shape {self.expected_shape}, got {signal.shape}")
        return signal

    def predict_raw(self, signal):
        signal = self.validate_input(signal)
        raw_prob = self.model.predict(signal[np.newaxis, ...], verbose=0).ravel()[0]
        return float(raw_prob)


class CalibrationEngine:
    """Wraps a fitted Platt scaler (ECG branch only).
    Clinical RF's CalibratedClassifierCV already outputs calibrated probability."""

    def __init__(self, platt_model):
        self.platt_model = platt_model

    def calibrate(self, raw_prob):
        return float(self.platt_model.predict_proba([[raw_prob]])[0, 1])


class FusionEngine:
    SEVERITY_BANDS = [
        (0.30, "Low"),
        (0.60, "Moderate"),
        (0.85, "High"),
        (1.01, "Critical")
    ]

    def __init__(self, fusion_fn):
        self.fusion_fn = fusion_fn

    def fuse(self, clinical_prob, ecg_prob):
        result = self.fusion_fn(clinical_prob, ecg_prob)
        fused_prob = result["fused_probability"] if isinstance(result, dict) else result
        return float(fused_prob)

    def get_severity_heuristic(self, fused_prob):
        # IMPORTANT: this is a probability-banded HEURISTIC, not a trained
        # severity model. Real multiclass severity prediction is planned
        # for Notebook 13 and will replace this field once available.
        for threshold, label in self.SEVERITY_BANDS:
            if fused_prob < threshold:
                return label
        return "Critical"


class ExplainabilityEngine:
    def __init__(self, rf_model, cnn_model, feature_order):
        self.shap_explainer = shap.TreeExplainer(rf_model)
        self.cnn_model = cnn_model
        self.feature_order = feature_order
        self.last_conv_layer = self._get_last_conv_layer(cnn_model)

    def _get_last_conv_layer(self, model):
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv1D):
                return layer.name
        return None

    def explain_clinical(self, patient_dict, top_k=5):
        X = pd.DataFrame([patient_dict])[self.feature_order]
        raw_shap = self.shap_explainer.shap_values(X)
        shap_vals = raw_shap[..., 1][0] if not isinstance(raw_shap, list) else raw_shap[1][0]

        contributions = sorted(
            zip(self.feature_order, shap_vals),
            key=lambda x: abs(x[1]), reverse=True
        )[:top_k]
        return [{"feature": f, "shap_value": round(float(v), 4)} for f, v in contributions]

    def explain_ecg(self, signal, top_k_leads=3, steps=20):
        baseline = np.zeros_like(signal)
        input_tensor = tf.convert_to_tensor(signal[np.newaxis, ...], dtype=tf.float32)
        baseline_tensor = tf.convert_to_tensor(baseline[np.newaxis, ...], dtype=tf.float32)

        alphas = tf.linspace(0.0, 1.0, steps)
        interpolated = tf.concat(
            [baseline_tensor + a * (input_tensor - baseline_tensor) for a in alphas], axis=0
        )

        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            preds = self.cnn_model(interpolated)
            loss = preds[:, 0]

        grads = tape.gradient(loss, interpolated)
        avg_grads = tf.reduce_mean(grads, axis=0)
        ig = (input_tensor[0] - baseline_tensor[0]) * avg_grads
        per_lead_attribution = tf.reduce_sum(tf.abs(ig), axis=0).numpy()

        lead_names = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
        ranked = sorted(zip(lead_names, per_lead_attribution), key=lambda x: x[1], reverse=True)[:top_k_leads]
        return [{"lead": l, "attribution": round(float(v), 6)} for l, v in ranked]


class RecommendationEngine:
    def generate(self, fused_prob, severity, quality_flags):
        recs = []

        if severity == "Low":
            recs.append("No significant cardiac risk indicators detected. Routine monitoring advised.")
        elif severity == "Moderate":
            recs.append("Some risk indicators present. Recommend follow-up with a cardiologist.")
        elif severity == "High":
            recs.append("Multiple risk indicators present. Prompt cardiology consultation recommended.")
        else:
            recs.append("Strong risk indicators detected. Urgent cardiology evaluation recommended.")

        if quality_flags:
            recs.append("Note: ECG signal quality issues detected — results may be less reliable. Consider re-recording.")

        recs.append("This is a screening tool, not a diagnostic substitute. Consult a licensed physician.")
        return recs


class CardioSensePipeline:
    def __init__(self, clinical_engine, ecg_engine, quality_engine,
                 calibration_engine, fusion_engine, explain_engine, recommendation_engine):
        self.clinical_engine = clinical_engine
        self.ecg_engine = ecg_engine
        self.quality_engine = quality_engine
        self.calibration_engine = calibration_engine
        self.fusion_engine = fusion_engine
        self.explain_engine = explain_engine
        self.recommendation_engine = recommendation_engine

    def run(self, patient_dict, ecg_signal):
        quality_report = self.quality_engine.assess(ecg_signal)

        clinical_raw = self.clinical_engine.predict_raw(patient_dict)
        # Clinical's CalibratedClassifierCV already outputs calibrated probability
        clinical_calibrated = clinical_raw

        ecg_raw = self.ecg_engine.predict_raw(ecg_signal)
        ecg_calibrated = self.calibration_engine.calibrate(ecg_raw)

        fused_prob = self.fusion_engine.fuse(clinical_calibrated, ecg_calibrated)
        severity = self.fusion_engine.get_severity_heuristic(fused_prob)

        clinical_contribution = abs(clinical_calibrated - 0.5)
        ecg_contribution = abs(ecg_calibrated - 0.5)
        total = clinical_contribution + ecg_contribution + 1e-8

        top_clinical_features = self.explain_engine.explain_clinical(patient_dict)
        top_ecg_leads = self.explain_engine.explain_ecg(ecg_signal)

        recommendations = self.recommendation_engine.generate(
            fused_prob, severity, quality_report["flags"]
        )

        return {
            "prediction": "Disease" if fused_prob >= 0.5 else "No Disease",
            "fused_probability": round(fused_prob, 4),
            "severity": severity,
            "severity_source": "heuristic_probability_band",
            "confidence": round(max(fused_prob, 1 - fused_prob), 4),
            "branch_contribution": {
                "clinical_pct": round(clinical_contribution / total * 100, 1),
                "ecg_pct": round(ecg_contribution / total * 100, 1)
            },
            "branch_probabilities": {
                "clinical": round(clinical_calibrated, 4),
                "ecg": round(ecg_calibrated, 4)
            },
            "top_clinical_features": top_clinical_features,
            "top_ecg_leads": top_ecg_leads,
            "ecg_quality": quality_report,
            "recommendations": recommendations,
            "disclaimer": "CardioSense AI is a screening aid, not a diagnostic tool. Always consult a licensed physician."
        }
