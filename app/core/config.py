from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database
    database_url: str = Field(
        default="sqlite+aiosqlite:///./dev.db",
        alias="DATABASE_URL",
        description="SQLAlchemy async DB URL (e.g. mysql+asyncmy://user:pass@host/db)",
    )

    # Auth
    # Provide a safe dev default so the module imports during tests; production should override.
    jwt_secret: SecretStr = Field(default=SecretStr("CHANGE_ME"), alias="JWT_SECRET")
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    access_token_exp_minutes: int = Field(default=60 * 24, alias="ACCESS_TOKEN_EXP_MINUTES")

    # Gemini
    gemini_api_key: SecretStr | None = Field(default=None, alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-3-flash-preview", alias="GEMINI_MODEL")

    # Uploads / persistence
    uploads_root: str = Field(default="uploads", alias="UPLOADS_ROOT")
    vectorstore_root: str = Field(default="vectorstore", alias="VECTORSTORE_ROOT")
    max_pdf_bytes: int = Field(default=10 * 1024 * 1024, alias="MAX_PDF_BYTES")  # 10MB default

    # Bootstrap/admin onboarding
    bootstrap_admin_email: str | None = Field(default=None, alias="BOOTSTRAP_ADMIN_EMAIL")
    admin_invite_token: str | None = Field(default=None, alias="ADMIN_INVITE_TOKEN")

    # App
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")


settings = Settings()

