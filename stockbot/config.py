from pydantic import BaseModel
import os
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

class Settings(BaseModel):
    # API keys
    newsapi_key: str | None = None
    finnhub_key: str | None = None
    polygon_key: str | None = None
    openai_api_key: str | None = None
    hf_api_token: str | None = None

    # App security/config
    secret_key: str = os.getenv("SECRET_KEY", "change-this-dev-key")
    jwt_secret: str | None = os.getenv("JWT_SECRET")
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./stockbot.db")

    # Feature flags
    deep_nlp_enabled: bool = os.getenv("DEEP_NLP_ENABLED", "true").lower() in ("1", "true", "yes")
    llm_enabled: bool = os.getenv("LLM_ENABLED", "false").lower() in ("1", "true", "yes")
    llm_model: str = os.getenv("LLM_MODEL", "google/flan-t5-base")

    # Timeouts/concurrency
    request_timeout_seconds: int = 240  # default overall time budget
    per_request_timeout_seconds: int = 20
    max_concurrency: int = 8
    llm_timeout_seconds: int = int(os.getenv("LLM_TIMEOUT_SECONDS", "18"))
    llm_max_input_chars: int = int(os.getenv("LLM_MAX_INPUT_CHARS", "4000"))

    @classmethod
    def load(cls) -> "Settings":
        return cls(
            newsapi_key=os.getenv("NEWSAPI_KEY"),
            finnhub_key=os.getenv("FINNHUB_KEY"),
            polygon_key=os.getenv("POLYGON_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            hf_api_token=os.getenv("HF_API_TOKEN"),
        )

