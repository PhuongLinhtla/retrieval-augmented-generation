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
    embedding_backend: str
    embedding_model: str
    ollama_host: str
    ollama_embedding_model: str
    ollama_llm_model: str
    ollama_fallback_model: str
    ollama_vision_model: str
    ollama_num_ctx: int
    ollama_temperature: float
    ollama_timeout: float
    chunk_size: int
    chunk_overlap: int
    parent_chunk_size: int
    top_k: int
    top_k_dense: int
    top_k_sparse: int
    fusion_top_k: int
    rerank_top_k: int
    ann_candidates: int
    min_similarity: float
    mmr_lambda: float
    use_hybrid_retrieval: bool
    use_reranker: bool
    reranker_model: str
    reranker_batch_size: int
    reranker_low_score_threshold: float
    low_confidence_threshold: float
    low_confidence_margin: float
    low_confidence_top_gap: float
    child_chunk_target_tokens: int
    child_chunk_min_tokens: int
    child_chunk_max_tokens: int
    child_chunk_overlap_sentences: int
    parent_chunk_target_tokens: int
    parent_chunk_min_tokens: int
    parent_chunk_max_tokens: int
    parent_child_group_min: int
    parent_child_group_max: int
    evidence_sentence_min: int
    evidence_sentence_max: int
    evidence_chunk_min: int
    evidence_chunk_max: int
    evidence_min_tokens: int
    evidence_max_tokens: int
    enable_evidence_compression: bool
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


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default

    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    return default


def load_settings(project_root: Path | None = None, llm_provider: str | None = None) -> Settings:
    root = project_root or Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env", override=False)

    documents_dir = root / "documents"
    storage_dir = root / "storage"
    storage_dir.mkdir(parents=True, exist_ok=True)
    documents_dir.mkdir(parents=True, exist_ok=True)

    selected_provider = (llm_provider or os.getenv("LLM_PROVIDER", "ollama")).strip().lower()
    if selected_provider not in {"local", "ollama", "openai", "gemini"}:
        selected_provider = "ollama"

    embedding_backend = os.getenv("EMBEDDING_BACKEND", "sentence_transformers").strip().lower()
    if embedding_backend not in {"sentence_transformers", "ollama"}:
        embedding_backend = "sentence_transformers"

    ollama_host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").strip().rstrip("/")
    if not ollama_host:
        ollama_host = "http://127.0.0.1:11434"

    ollama_llm_model = os.getenv("OLLAMA_LLM_MODEL", "qwen2.5:14b-instruct-q4_K_M").strip()
    if not ollama_llm_model:
        ollama_llm_model = "qwen2.5:14b-instruct-q4_K_M"

    ollama_fallback_model = os.getenv(
        "OLLAMA_FALLBACK_MODEL",
        "llama3.1:8b-instruct-q5_K_M",
    ).strip()

    ollama_vision_model = os.getenv("OLLAMA_VISION_MODEL", "moondream").strip()
    if not ollama_vision_model:
        ollama_vision_model = "moondream"

    ollama_embedding_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text").strip()
    if not ollama_embedding_model:
        ollama_embedding_model = "nomic-embed-text"

    ollama_num_ctx = max(1024, _int_env("OLLAMA_NUM_CTX", 8192))
    ollama_temperature = min(0.2, max(0.0, _float_env("OLLAMA_TEMPERATURE", 0.1)))
    ollama_timeout = max(10.0, _float_env("OLLAMA_TIMEOUT", 120.0))

    chunk_size = max(220, _int_env("CHUNK_SIZE", 750))
    chunk_overlap = max(30, _int_env("CHUNK_OVERLAP", 120))
    parent_chunk_size = max(
        chunk_size + 120,
        _int_env("PARENT_CHUNK_SIZE", max(1400, chunk_size * 3)),
    )

    child_chunk_target_tokens = max(220, _int_env("CHILD_CHUNK_TARGET_TOKENS", 450))
    child_chunk_min_tokens = max(120, _int_env("CHILD_CHUNK_MIN_TOKENS", 200))
    child_chunk_max_tokens = max(
        child_chunk_target_tokens,
        _int_env("CHILD_CHUNK_MAX_TOKENS", 700),
    )

    parent_chunk_target_tokens = max(600, _int_env("PARENT_CHUNK_TARGET_TOKENS", 1200))
    parent_chunk_min_tokens = max(350, _int_env("PARENT_CHUNK_MIN_TOKENS", 900))
    parent_chunk_max_tokens = max(
        parent_chunk_target_tokens,
        _int_env("PARENT_CHUNK_MAX_TOKENS", 1600),
    )

    return Settings(
        project_root=root,
        documents_dir=documents_dir,
        storage_dir=storage_dir,
        db_path=storage_dir / "rag_metadata.sqlite3",
        index_path=storage_dir / "rag_hnsw.index",
        index_meta_path=storage_dir / "rag_hnsw_meta.json",
        embedding_backend=embedding_backend,
        embedding_model=os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ),
        ollama_host=ollama_host,
        ollama_embedding_model=ollama_embedding_model,
        ollama_llm_model=ollama_llm_model,
        ollama_fallback_model=ollama_fallback_model,
        ollama_vision_model=ollama_vision_model,
        ollama_num_ctx=ollama_num_ctx,
        ollama_temperature=ollama_temperature,
        ollama_timeout=ollama_timeout,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        parent_chunk_size=parent_chunk_size,
        top_k=max(1, _int_env("TOP_K", 3)),
        top_k_dense=max(4, _int_env("TOP_K_DENSE", 48)),
        top_k_sparse=max(4, _int_env("TOP_K_SPARSE", 48)),
        fusion_top_k=max(12, _int_env("FUSION_TOP_K", 96)),
        rerank_top_k=max(4, _int_env("RERANK_TOP_K", 10)),
        ann_candidates=max(3, _int_env("ANN_CANDIDATES", 12)),
        min_similarity=min(0.99, max(0.0, _float_env("MIN_SIMILARITY", 0.30))),
        mmr_lambda=min(0.95, max(0.1, _float_env("MMR_LAMBDA", 0.68))),
        use_hybrid_retrieval=_bool_env("USE_HYBRID_RETRIEVAL", True),
        use_reranker=_bool_env("USE_RERANKER", True),
        reranker_model=os.getenv(
            "RERANKER_MODEL",
            "BAAI/bge-reranker-v2-m3",
        ).strip(),
        reranker_batch_size=max(2, _int_env("RERANKER_BATCH_SIZE", 8)),
        reranker_low_score_threshold=min(
            1.0,
            max(0.0, _float_env("RERANKER_LOW_SCORE_THRESHOLD", 0.34)),
        ),
        low_confidence_threshold=min(
            1.0,
            max(0.0, _float_env("LOW_CONFIDENCE_THRESHOLD", 0.46)),
        ),
        low_confidence_margin=min(1.0, max(0.0, _float_env("LOW_CONFIDENCE_MARGIN", 0.06))),
        low_confidence_top_gap=min(1.0, max(0.0, _float_env("LOW_CONFIDENCE_TOP_GAP", 0.04))),
        child_chunk_target_tokens=child_chunk_target_tokens,
        child_chunk_min_tokens=child_chunk_min_tokens,
        child_chunk_max_tokens=child_chunk_max_tokens,
        child_chunk_overlap_sentences=max(1, _int_env("CHILD_CHUNK_OVERLAP_SENTENCES", 2)),
        parent_chunk_target_tokens=parent_chunk_target_tokens,
        parent_chunk_min_tokens=parent_chunk_min_tokens,
        parent_chunk_max_tokens=parent_chunk_max_tokens,
        parent_child_group_min=max(2, _int_env("PARENT_CHILD_GROUP_MIN", 2)),
        parent_child_group_max=max(2, _int_env("PARENT_CHILD_GROUP_MAX", 4)),
        evidence_sentence_min=max(8, _int_env("EVIDENCE_SENTENCE_MIN", 15)),
        evidence_sentence_max=max(10, _int_env("EVIDENCE_SENTENCE_MAX", 40)),
        evidence_chunk_min=max(1, _int_env("EVIDENCE_CHUNK_MIN", 6)),
        evidence_chunk_max=max(1, _int_env("EVIDENCE_CHUNK_MAX", 10)),
        evidence_min_tokens=max(300, _int_env("EVIDENCE_MIN_TOKENS", 1500)),
        evidence_max_tokens=max(600, _int_env("EVIDENCE_MAX_TOKENS", 3000)),
        enable_evidence_compression=_bool_env("ENABLE_EVIDENCE_COMPRESSION", True),
        llm_provider=selected_provider,
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip(),
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip(),
    )
