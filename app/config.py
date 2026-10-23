"""Application settings loaded from environment variables."""
from dataclasses import dataclass
import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # .env loading is optional at runtime
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _bool(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    app_name: str = "Autonomous Document Agent"

    # LLM providers: "ollama" (local) or "openai" (any OpenAI-compatible API: Groq, Gemini, LM Studio...)
    llm_provider: str = "ollama"
    llm_fallback_provider: str = ""  # optional second provider used when the primary fails
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    openai_base_url: str = "https://api.groq.com/openai/v1"
    openai_api_key: str = ""
    gemini_api_key: str = ""
    openai_model: str = "gemini-3.8-flash"
    llm_timeout_seconds: float = 180.0
    llm_max_retries: int = 2
    llm_backoff_seconds: float = 1.0

    # Agent behaviour
    max_concurrency: int = 1  # keep local LLM inference serial on memory-constrained machines
    task_max_attempts: int = 2
    max_revision_rounds: int = 1
    min_request_chars: int = 15
    max_request_chars: int = 5000

    output_dir: Path = PROJECT_ROOT / "output" / "documents"
    include_document_base64: bool = True

    @classmethod
    def from_env(cls) -> "Settings":
        e = os.getenv
        return cls(
            app_name=e("APP_NAME", cls.app_name),
            llm_provider=e("LLM_PROVIDER", "ollama").strip().lower(),
            llm_fallback_provider=e("LLM_FALLBACK_PROVIDER", "").strip().lower(),
            ollama_base_url=e("OLLAMA_BASE_URL", cls.ollama_base_url).rstrip("/"),
            ollama_model=e("OLLAMA_MODEL", cls.ollama_model),
            openai_base_url=e("OPENAI_BASE_URL", cls.openai_base_url).rstrip("/"),
            openai_api_key=e("OPENAI_API_KEY", ""),
            gemini_api_key=e("GEMINI_API_KEY", ""),
            openai_model=e("OPENAI_MODEL", cls.openai_model),
            llm_timeout_seconds=float(e("LLM_TIMEOUT_SECONDS", "180")),
            llm_max_retries=int(e("LLM_MAX_RETRIES", "2")),
            llm_backoff_seconds=float(e("LLM_BACKOFF_SECONDS", "1.0")),
            max_concurrency=int(e("MAX_CONCURRENCY", "1")),
            task_max_attempts=int(e("TASK_MAX_ATTEMPTS", "2")),
            max_revision_rounds=int(e("MAX_REVISION_ROUNDS", "1")),
            min_request_chars=int(e("MIN_REQUEST_CHARS", "15")),
            max_request_chars=int(e("MAX_REQUEST_CHARS", "5000")),
            output_dir=Path(e("OUTPUT_DIR", str(cls.output_dir))).resolve(),
            include_document_base64=_bool("INCLUDE_DOCUMENT_BASE64", "true"),
        )
