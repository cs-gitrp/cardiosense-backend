import uuid
from datetime import datetime

from sqlalchemy import Column, String, DateTime, Boolean
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from app.db.session import Base


class ModelRun(Base):
    """
    One row per locked model artifact (e.g. Notebook 09 fusion pkl,
    Notebook 12 inference pipeline version, Notebook 13 severity model).
    Powers GET /insights/model-comparison and GET /insights/ablation,
    and lets Assessment rows point at exactly which model version
    produced them (model_version / git_commit_hash — added per the
    Phase 7 discussion, since the CNN-LSTM -> CNN-only and heuristic ->
    trained-severity swaps already happened once and will happen again).
    """
    __tablename__ = "model_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    component = Column(String, nullable=False)     # "clinical_branch" | "ecg_branch" | "fusion" | "severity"
    model_version = Column(String, nullable=False)  # e.g. "13_severity_rf_v1"
    git_commit_hash = Column(String, nullable=True)
    metrics = Column(JSONB, nullable=True)           # full metric dict for this run (Section 5.4)
    is_active = Column(Boolean, default=False)        # which version is currently live in /assess
    notes = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    assessments = relationship("Assessment", back_populates="model_run")
