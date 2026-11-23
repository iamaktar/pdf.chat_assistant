"""Configuration management and environment variable validation."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Provider selection ("gemini" or "openai")
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini").strip().lower()

# API Keys
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "").strip()
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "").strip()

# Model Names
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_EMBEDDING_MODEL: str = os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")

OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# RAG Hyperparameters
CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 200
TOP_K_RESULTS: int = 4
TEMPERATURE: float = 0.1

# Storage Path (pure-Python JSON vector store)
VECTOR_STORE_FILE: Path = Path("vector_store.json")


def validate_config() -> tuple[bool, str]:
    """Validate that the selected LLM provider and its API key are configured."""
    if LLM_PROVIDER not in {"gemini", "openai"}:
        return (
            False,
            f"Unsupported LLM_PROVIDER '{LLM_PROVIDER}'. Use 'gemini' or 'openai' in your .env file.",
        )

    if LLM_PROVIDER == "gemini" and not GOOGLE_API_KEY:
        return False, "GOOGLE_API_KEY is missing. Add it to your .env file."

    if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
        return False, "OPENAI_API_KEY is missing. Add it to your .env file."

    return True, ""