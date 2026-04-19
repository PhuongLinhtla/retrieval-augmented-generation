from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


@dataclass(frozen=True, slots=True)
class Settings:
    project_root: Path
    documents_dir: Path
    storage_dir: Path
    db_path: Path
    index_path: Path
    index_meta_path: Path
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    ann_candidates: int
    min_similarity: float
    mmr_lambda: float
    llm_provider: str
    openai_api_key: str
    openai_model: str
    gemini_api_key: str
    gemini_model: str


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def load_settings(project_root: Path | None = None, llm_provider: str | None = None) -> Settings:
    root = project_root or Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env", override=False)

    documents_dir = root / "documents"
    storage_dir = root / "storage"
    storage_dir.mkdir(parents=True, exist_ok=True)
    documents_dir.mkdir(parents=True, exist_ok=True)

    selected_provider = (llm_provider or os.getenv("LLM_PROVIDER", "local")).strip().lower()
    if selected_provider not in {"local", "openai", "gemini"}:
        selected_provider = "local"

    return Settings(
        project_root=root,
        documents_dir=documents_dir,
        storage_dir=storage_dir,
        db_path=storage_dir / "rag_metadata.sqlite3",
        index_path=storage_dir / "rag_hnsw.index",
        index_meta_path=storage_dir / "rag_hnsw_meta.json",
        embedding_model=os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ),
        chunk_size=max(250, _int_env("CHUNK_SIZE", 750)),
        chunk_overlap=max(30, _int_env("CHUNK_OVERLAP", 120)),
        top_k=max(1, _int_env("TOP_K", 3)),
        ann_candidates=max(3, _int_env("ANN_CANDIDATES", 12)),
        min_similarity=min(0.99, max(0.0, _float_env("MIN_SIMILARITY", 0.30))),
        mmr_lambda=min(0.95, max(0.1, _float_env("MMR_LAMBDA", 0.68))),
        llm_provider=selected_provider,
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip(),
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip(),
    )
