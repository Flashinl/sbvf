from pydantic import BaseModel
import os
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

class Settings(BaseModel):
    newsapi_key: str | None = None
    finnhub_key: str | None = None
    polygon_key: str | None = None
    openai_api_key: str | None = None
    request_timeout_seconds: int = 240  # default overall time budget
    per_request_timeout_seconds: int = 20
    max_concurrency: int = 8

    @classmethod
    def load(cls) -> "Settings":
        return cls(
            newsapi_key=os.getenv("NEWSAPI_KEY"),
            finnhub_key=os.getenv("FINNHUB_KEY"),
            polygon_key=os.getenv("POLYGON_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

