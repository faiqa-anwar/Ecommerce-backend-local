"""
Central app configuration. Everything here is overridable via environment
variables or a .env file (see .env.example).
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Database ---
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/ecommerce_db"

    # --- Auth ---
    # IMPORTANT: change this. Used to sign JWTs and password reset tokens.
    SECRET_KEY: str = "change-me-in-production-please"
    ACCESS_TOKEN_LIFETIME_SECONDS: int = 3600

    # --- Local LLM (Ollama by default) ---
    # Ollama exposes an OpenAI-compatible API at /v1, so the standard openai
    # SDK works as-is. Any other OpenAI-compatible server also works (LM Studio,
    # GPT4All's API server, llama.cpp server, ...) -- just change the base URL.
    LLM_BASE_URL: str = "http://localhost:11434/v1"
    LLM_API_KEY: str = "ollama"  # Ollama ignores it, but the SDK requires a non-empty value
    # Must be a model you've pulled (`ollama pull llama3.2`) and one that
    # supports tool calling for order/inventory lookups to work.
    LLM_MODEL: str = "llama3.2"
    LLM_MAX_TOOL_ITERATIONS: int = 5
    # Local models on CPU can be slow -- be generous.
    LLM_REQUEST_TIMEOUT_SECONDS: float = 180.0

    # --- Embeddings ---
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384

    # --- RAG ---
    CHUNK_SIZE_CHARS: int = 500
    CHUNK_OVERLAP_CHARS: int = 50
    RAG_TOP_K: int = 4

    # --- Chat memory ---
    MAX_HISTORY_MESSAGES: int = 12  # how many past turns to feed back to the LLM

    # --- Rate limiting (simple in-memory limiter on the public /chat endpoint) ---
    CHAT_RATE_LIMIT_PER_MINUTE: int = 20


settings = Settings()
