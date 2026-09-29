"""Central configuration, read from the environment / .env file."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


# --- LLM (Llama) -------------------------------------------------------------
# "ollama" = run Llama locally (default), "groq" = hosted Llama via Groq API.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")

_is_ollama = LLM_PROVIDER == "ollama"
# General model: chat, summaries, Hindi translation.
LLM_MODEL = os.getenv("LLM_MODEL") or ("llama3.1:8b" if _is_ollama else "llama-3.3-70b-versatile")
# Specific model for prescription extraction + medical explanations.
# Point this at a medical fine-tune (e.g. a Llama-based medical model you pulled
# into Ollama) — defaults to the general Llama model.
MEDICAL_MODEL = os.getenv("MEDICAL_MODEL") or LLM_MODEL
# Vision model used to read photos / scanned prescriptions.
VISION_MODEL = os.getenv("VISION_MODEL") or (
    "llama3.2-vision" if _is_ollama else "meta-llama/llama-4-scout-17b-16e-instruct"
)
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "300"))
LLM_NUM_CTX = int(os.getenv("LLM_NUM_CTX", "12288"))
# Ollama processes one request at a time by default; Groq can take several.
LLM_PARALLELISM = int(os.getenv("LLM_PARALLELISM", "1" if _is_ollama else "4"))

# --- Knowledge base (RAG, like the reference repo) ---------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
VECTOR_STORE = os.getenv("VECTOR_STORE", "faiss").strip().lower()  # faiss | pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medical-chatbot")
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", BASE_DIR / "vectorstore"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))

# --- Trusted online sources (openFDA, RxNorm, MedlinePlus) --------------------
ENABLE_ONLINE_SOURCES = _bool("ENABLE_ONLINE_SOURCES", True)
OPENFDA_API_KEY = os.getenv("OPENFDA_API_KEY", "")
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "15"))

# --- App limits ---------------------------------------------------------------
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "15"))
MAX_ITEMS = int(os.getenv("MAX_ITEMS", "12"))
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "200"))
