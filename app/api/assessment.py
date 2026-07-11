# pyrefly: ignore [missing-import]
from fastapi import APIRouter, Depends, HTTPException, status
# pyrefly: ignore [missing-import]
from fastapi.responses import StreamingResponse
# pyrefly: ignore [missing-import]
from sqlalchemy.orm import Session
from datetime import timedelta

from app.api.deps import get_current_user, get_db
from app.models.user import User
from app.models.assessment import Assessment
from app.schemas.assessment import AssessRequest, AssessResponse, AssessmentHistoryItem, BranchContribution, BranchProbabilities
from app.services.assessment_service import run_assessment, run_assessment_generator, _compile_prompt_summaries

router = APIRouter(prefix="/assess", tags=["assessment"])


@router.post("", response_model=AssessResponse, status_code=status.HTTP_201_CREATED)
def create_assessment(
    payload: AssessRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        return run_assessment(payload, str(current_user.id), db)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))


@router.post("/run-stream")
def run_assessment_stream(
    payload: AssessRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    def event_generator():
        try:
            for event in run_assessment_generator(payload, str(current_user.id), db):
                yield f"{event}\n"
        except Exception as e:
            yield f"ERROR:{str(e)}\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/history", response_model=list[AssessmentHistoryItem])
def get_assessment_history(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    limit: int = 20,
    offset: int = 0,
):
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
            created_at=(r.created_at + timedelta(hours=5, minutes=30)).isoformat() if r.created_at else None,
        )
        for r in rows
    ]


@router.get("/{assessment_id}", response_model=AssessResponse)
def get_assessment(
    assessment_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    row = (
        db.query(Assessment)
        .filter(Assessment.id == assessment_id, Assessment.user_id == current_user.id)
        .first()
    )
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Assessment not found.")

    # Generate the missing text fields on the fly during data extraction
    enriched = _compile_prompt_summaries(
        row.fused_probability,
        row.prediction,
        row.severity,
        row.severity_source,
        row.clinical_input or {},
        row.top_ecg_leads
    )

    bc = row.branch_contribution
    bp = row.branch_probabilities

    return AssessResponse(
        assessment_id=str(row.id),
        prediction=row.prediction,
        fused_probability=row.fused_probability,
        severity=row.severity,
        severity_source=row.severity_source,
        confidence=row.confidence,
        branch_contribution=BranchContribution(**bc) if bc else None,
        branch_probabilities=BranchProbabilities(**bp) if bp else None,
        top_clinical_features=row.top_clinical_features,
        top_ecg_leads=row.top_ecg_leads,
        ecg_quality=row.ecg_quality,
        recommendations=row.recommendations,
        feature_missingness_map=row.feature_missingness_map,
        disclaimer=row.disclaimer or "",
        clinical=row.clinical_input or {},
        created_at=row.created_at.isoformat() if row.created_at else None,
        **enriched
    )