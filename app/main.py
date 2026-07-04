from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from app.api import auth, assessment, insights, chat
from app.api.deps import get_current_user
from app.core.config import settings
from app.db.session import Base, engine
from app.models import User  # noqa: F401 — ensures all models registered before create_all
from app.core.config import settings

print("="*60)
print("GROQ =", settings.GROQ_API_KEY)
print("="*60)

# ---------------------------------------------------------------------------
# Startup: create tables + warm up ML models
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create tables (use Alembic in production; this is safe for dev)
    Base.metadata.create_all(bind=engine)

    # Warm up model pipeline so the first /assess request isn't slow
    if settings.ENVIRONMENT != "test":
        try:
            from app.services.model_loader import get_pipeline, get_severity_model
            get_pipeline()
            get_severity_model()
            print("CardioSense ML artifacts loaded successfully.")
        except FileNotFoundError as e:
            print(f"WARNING: ML artifacts not found — /assess will fail until loaded. {e}")

    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CardioSense AI API",
    description=(
        "Multimodal heart disease prediction API. "
        "Confidence-adaptive fusion of clinical (tabular) and ECG branches. "
        "NOT a medical device. Research and academic use only."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://cardiosense.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

# Auth — /auth/register, /auth/login
app.include_router(auth.router)

# Assessment — /assess (POST), /assess/history, /assess/{id}
# /auth/me needs the real dependency, fix the placeholder in auth.py
app.include_router(
    assessment.router,
    dependencies=[],  # per-route deps in assessment.py itself
)

# Insights — /insights/calibration, /model-comparison, /bootstrap-ci
app.include_router(insights.router)
app.include_router(chat.router)


# Fix /auth/me to use the real dependency (avoids circular import in auth.py)
from fastapi import APIRouter
me_router = APIRouter()

@me_router.get("/auth/me")
def me(current_user: User = Depends(get_current_user)):
    return {"id": str(current_user.id), "email": current_user.email, "full_name": current_user.full_name}

app.include_router(me_router)


@app.get("/health")
def health():
    return {"status": "ok", "environment": settings.ENVIRONMENT}
