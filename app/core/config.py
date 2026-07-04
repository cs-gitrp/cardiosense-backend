from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 10080  # 7 days
    MODEL_DIR: str = "./ml_artifacts"
    ENVIRONMENT: str = "development"
    GROQ_API_KEY: str | None = None
    BACKEND_CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "https://cardiosense.vercel.app",
    ]

    class Config:
        env_file = ".env"


settings = Settings()
