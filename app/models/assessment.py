import uuid
from datetime import datetime

from sqlalchemy import Column, String, DateTime, Float, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from app.db.session import Base


class Assessment(Base):
    """
    Mirrors the exact return dict of pipeline.run() from Notebook 12, plus
    two Phase 7 additions:
      - feature_missingness_map: which of the 11 clinical features were
        missing/imputed for this patient (turns the Notebook 03-08
        "missingness as a feature" novelty into something queryable
        post-deployment, not just an illustrative notebook case study).
      - model_version: FK to model_runs, so retrains (e.g. once the
        Notebook 13 severity model replaces the heuristic) don't silently
        blend old and new predictions in the same history view.
    """
    __tablename__ = "assessments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    model_run_id = Column(UUID(as_uuid=True), ForeignKey("model_runs.id"), nullable=True)

    # --- Inputs ---
    clinical_input = Column(JSONB, nullable=False)          # raw patient_dict as submitted
    feature_missingness_map = Column(JSONB, nullable=True)  # {"ca": True, "thal": False, ...}
    ecg_quality = Column(JSONB, nullable=True)               # quality_report from pipeline output

    # --- Outputs (mirrors pipeline.run() return dict exactly) ---
    prediction = Column(String, nullable=False)               # "Disease" / "No Disease"
    fused_probability = Column(Float, nullable=False)
    severity = Column(String, nullable=True)   # "Low"|"Moderate"|"High"|"Critical"
    severity_source = Column(String, nullable=True)            # "heuristic_probability_band" |
                                                                 # "trained_multiclass_rf" | "trained_multiclass_xgb"
    confidence = Column(Float, nullable=True)
    branch_contribution = Column(JSONB, nullable=True)         # {"clinical_pct": .., "ecg_pct": ..}
    branch_probabilities = Column(JSONB, nullable=True)        # {"clinical": .., "ecg": ..}
    top_clinical_features = Column(JSONB, nullable=True)
    top_ecg_leads = Column(JSONB, nullable=True)
    recommendations = Column(JSONB, nullable=True)
    disclaimer = Column(String, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="assessments")
    model_run = relationship("ModelRun", back_populates="assessments")
