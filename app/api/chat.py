"""
CardioBot — RAG-powered clinical decision support chatbot.

Architecture (mirrors DocuMind):
  - System prompt: cardiology domain expert persona + disclaimer
  - Optional assessment context: if ?assessment_id=xxx is passed, the full
    assessment result is injected into the system prompt so the bot can
    reason about the specific patient's SHAP values, branch probabilities,
    ECG leads, and recommendations.
  - Groq (llama-3.1-8b-instant) for fast inference
  - No FAISS in this version — pure LLM with structured context injection.
    FAISS retrieval over clinical guidelines can be added in Sprint 6.

Environment variables required:
  GROQ_API_KEY = your Groq API key
"""

import os
import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.core.config import settings

from app.api.deps import get_current_user, get_db
from app.models.user import User
from app.models.assessment import Assessment
from app.core.config import settings

router = APIRouter(prefix="/chat", tags=["cardiobot"])

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"

CARDIOBOT_SYSTEM = """You are CardioBot, a conversational clinical support assistant embedded in the CardioSense AI platform. Adopt a hybrid persona that blends empathetic medical colleague tone with clinician-grade precision. Recognize casual greetings like "hii" or "hello", conversational queries, and polite chat closures like "thanks" or "goodbye". Respond naturally and fluidly instead of returning raw diagnostic blocks every turn.

When an assessment context is active, remain strictly grounded in the active patient's parsed SHAP values, branch probabilities, branch weights, ECG lead attributions, and model outputs. Do not invent arbitrary blood tests, lab values, medication doses, or patient data not provided in the assessment context.

When appropriate, offer evidence-based cardiovascular precautions, lifestyle adjustments, stress mitigation strategies, and dietary guidance such as the DASH diet or sodium restriction. Tailor your recommendations to the patient risk profile: use higher caution and proactive prevention for "Disease" or high-risk predictions, and balanced reassurance plus follow-up planning for lower-risk contexts.

Your role:
- Explain predictions, SHAP attributions, ECG lead attributions, branch probabilities, and confidence scores in clinician-friendly language.
- Reference ACC/AHA and other relevant cardiology guidance when appropriate.
- Keep responses natural, polite, and concise unless the user asks for more detail.
- Always remind users this is a research screening tool, not a diagnostic device.
- Use markdown formatting.
- Never fabricate lab values, medication doses, or patient data not provided.

{assessment_context}

IMPORTANT DISCLAIMER: CardioSense AI is a research prototype. It is NOT a medical device. All outputs require clinical validation by a licensed physician before any action is taken."""

FEATURE_LABELS = {
    "cp": "Chest Pain Type", "exang": "Exercise Angina", "ca": "Fluoroscopy Vessels",
    "sex": "Sex", "chol_missing": "Cholesterol Missing", "slope_missing": "ST Slope Missing",
    "thal": "Thalassemia", "thalach": "Max Heart Rate", "age": "Age",
    "oldpeak": "ST Depression (Oldpeak)", "fbs": "Fasting Blood Sugar"
}


class ChatMessage(BaseModel):
    role: str   # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    assessment_id: Optional[str] = None


def _build_assessment_context(assessment: Assessment) -> str:
    if not assessment:
        return ""

    lines = [
        f"\n## Active Patient Context (Assessment {str(assessment.id)[:8]}...)\n",
        f"- **Prediction**: {assessment.prediction}",
        f"- **Fused Probability**: {assessment.fused_probability:.1%}",
        f"- **Severity**: {assessment.severity}",
        f"- **Confidence**: {assessment.confidence:.1%}",
    ]

    if assessment.branch_probabilities:
        bp = assessment.branch_probabilities
        lines.append(f"- **Branch Probabilities**: Clinical {bp.get('clinical', 0):.1%} | ECG {bp.get('ecg', 'N/A')}")

    if assessment.branch_contribution:
        bc = assessment.branch_contribution
        lines.append(f"- **Branch Weights**: Clinical {bc.get('clinical_pct', 0):.0f}% | ECG {bc.get('ecg_pct', 0):.0f}%")

    if assessment.top_clinical_features:
        lines.append("\n**Top SHAP Features (Clinical RF)**:")
        for f in assessment.top_clinical_features[:5]:
            label = FEATURE_LABELS.get(f["feature"], f["feature"])
            direction = "↑ risk" if f["shap_value"] > 0 else "↓ risk"
            lines.append(f"  - {label}: {f['shap_value']:+.4f} ({direction})")

    if assessment.top_ecg_leads:
        lines.append("\n**Top ECG Lead Attributions (Integrated Gradients)**:")
        for lead in assessment.top_ecg_leads[:3]:
            lines.append(f"  - Lead {lead['lead']}: attribution {lead['attribution']:.4f}")

    if assessment.clinical_input:
        ci = assessment.clinical_input
        lines.append(f"\n**Clinical Input Summary**: Age {ci.get('age')}, "
                     f"{'Male' if ci.get('sex') == 1 else 'Female'}, "
                     f"CP Type {ci.get('cp')}, "
                     f"Max HR {ci.get('thalach')} bpm, "
                     f"Oldpeak {ci.get('oldpeak')} mm")

    if assessment.recommendations:
        lines.append("\n**Model Recommendations**:")
        for r in assessment.recommendations[:2]:
            lines.append(f"  - {r}")

    return "\n".join(lines)


@router.post("")
async def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    CardioBot chat endpoint with optional assessment context injection.
    Streams the Groq response back to the frontend.
    """
    
    GROQ_API_KEY = (
        os.environ.get("GROQ_API_KEY") or 
        getattr(settings, "GROQ_API_KEY", "") or 
        getattr(settings, "groq_api_key", "")
        
    )
    
    print(f"DEBUG — OS Env: {bool(os.environ.get('GROQ_API_KEY'))} | Pydantic: {bool(getattr(settings, 'GROQ_API_KEY', ''))}")
    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GROQ_API_KEY not configured. Add it to your .env file."
        )

    # Load assessment context if provided
    assessment_context = ""
    if request.assessment_id:
        assessment = (
            db.query(Assessment)
            .filter(
                Assessment.id == request.assessment_id,
                Assessment.user_id == current_user.id
            )
            .first()
        )
        if assessment:
            assessment_context = _build_assessment_context(assessment)

    system_prompt = CARDIOBOT_SYSTEM.format(
        assessment_context=assessment_context if assessment_context
        else "\nNo active patient assessment loaded. Respond to general cardiology questions."
    )

    # Build message list for Groq
    groq_messages = [{"role": "system", "content": system_prompt}]
    for msg in request.messages[-12:]:  # Keep last 12 turns to stay within context
        groq_messages.append({"role": msg.role, "content": msg.content})

    # Call Groq with streaming
    import httpx

    async def stream_groq():
        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.stream(
                "POST",
                GROQ_URL,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": GROQ_MODEL,
                    "messages": groq_messages,
                    "stream": True,
                    "max_tokens": 600,
                    "temperature": 0.6,
                },
            ) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    yield f"data: {json.dumps({'error': f'Groq error {response.status_code}'})}\n\n"
                    return

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        yield "data: [DONE]\n\n"
                        return
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield f"data: {json.dumps({'content': delta})}\n\n"
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

    return StreamingResponse(
        stream_groq(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
