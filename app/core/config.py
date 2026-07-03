from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 10080  # 7 days
    MODEL_DIR: str = "./ml_artifacts"
    ENVIRONMENT: str = "development"

    class Config:
        env_file = ".env"


settings = Settings()
