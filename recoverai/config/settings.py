"""Configuration management using Pydantic Settings."""

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application configuration settings."""
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # General
    PROJECT_NAME: str = "RecoverAI"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"

    # Paths
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = BASE_DIR / "data"
    ARTIFACTS_DIR: Path = BASE_DIR / "artifacts"
    MODELS_DIR: Path = BASE_DIR / "artifacts" / "models"
    LOGS_DIR: Path = BASE_DIR / "logs"

    # API & Dashboard
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000
    DASHBOARD_PORT: int = 8501

    # Database
    DATABASE_URL: str = "sqlite:///./recoverai.db"

    # LLM Settings
    OPENAI_API_KEY: str = Field(default="", validation_alias="OPENAI_API_KEY")
    GEMINI_API_KEY: str = Field(default="", validation_alias="GEMINI_API_KEY")
    LLM_PROVIDER: str = "mock_fallback"  # Options: openai, gemini, mock_fallback
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.2

    # Safety Guardrails & Business Invariants
    MAX_RECOVERY_ATTEMPTS: int = 3
    COMMUNICATION_COOLDOWN_HOURS: int = 24
    HIGH_VALUE_THRESHOLD_INR: float = 50000.0
    MIN_EXPECTED_VALUE_INR: float = 5.0
    DEFAULT_EXPONENTIAL_BACKOFF_HOURS: list[int] = [1, 6, 24]

    # Unit Economic Costs (in INR)
    COST_AUTO_RETRY_INR: float = 0.50
    COST_WHATSAPP_INR: float = 0.80
    COST_SMS_INR: float = 0.25
    COST_EMAIL_INR: float = 0.05
    COST_UPI_INTENT_INR: float = 0.40
    COST_MANUAL_ESCALATION_INR: float = 50.00
    CUSTOMER_FRICTION_PENALTY_INR: float = 3.00

    def ensure_directories(self) -> None:
        """Ensure all required runtime directories exist."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        (self.DATA_DIR / "raw").mkdir(parents=True, exist_ok=True)
        (self.DATA_DIR / "processed").mkdir(parents=True, exist_ok=True)
        self.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.LOGS_DIR.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()
