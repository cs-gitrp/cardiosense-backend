from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db
from app.models.user import User
from app.models.assessment import Assessment
from app.schemas.assessment import AssessRequest, AssessResponse, AssessmentHistoryItem
from app.services.assessment_service import run_assessment

router = APIRouter(prefix="/assess", tags=["assessment"])


@router.post("", response_model=AssessResponse, status_code=status.HTTP_201_CREATED)
def create_assessment(
    payload: AssessRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Main inference endpoint.
    Accepts 11 clinical features + pre-processed ECG signal (flat 12000 floats).
    Returns the full pipeline.run() output plus severity gating and audit fields.
    """
    try:
        return run_assessment(payload, str(current_user.id), db)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))


@router.get("/history", response_model=list[AssessmentHistoryItem])
def get_assessment_history(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    limit: int = 20,
    offset: int = 0,
):
    """
    Returns the current user's assessment history, sorted newest first.
    Used for the Risk Trend graph in the History page (process.md Section 7).
    """
    rows = (
        db.query(Assessment)
        .filter(Assessment.user_id == current_user.id)
        .order_by(Assessment.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return [
        AssessmentHistoryItem(
            assessment_id=str(r.id),
            prediction=r.prediction,
            fused_probability=r.fused_probability,
            severity=r.severity,
            created_at=r.created_at.isoformat(),
        )
        for r in rows
    ]


@router.get("/{assessment_id}", response_model=AssessResponse)
def get_assessment(
    assessment_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Retrieves a single past assessment by ID.
    Used when the user revisits a previous result page.
    """
    row = (
        db.query(Assessment)
        .filter(Assessment.id == assessment_id, Assessment.user_id == current_user.id)
        .first()
    )
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Assessment not found.")

    return AssessResponse(
        assessment_id=str(row.id),
        prediction=row.prediction,
        fused_probability=row.fused_probability,
        severity=row.severity,
        severity_source=row.severity_source,
        confidence=row.confidence,
        branch_contribution=row.branch_contribution,
        branch_probabilities=row.branch_probabilities,
        top_clinical_features=row.top_clinical_features,
        top_ecg_leads=row.top_ecg_leads,
        ecg_quality=row.ecg_quality,
        recommendations=row.recommendations,
        feature_missingness_map=row.feature_missingness_map,
        disclaimer=row.disclaimer or "",
    )
